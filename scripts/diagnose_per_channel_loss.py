#!/usr/bin/env python3
"""Run one training batch and print per-variable loss breakdown.

Useful when total batch_loss is high but the scalar does not identify which
output channels dominate. Uses the same loss path as training (normalization,
weights, n_forward_steps) with NullOptimization so weights are not updated.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

_WORKTREE = Path(__file__).resolve().parents[1]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))
os.environ.setdefault("PYTHONPATH", str(_WORKTREE))

import dacite
import torch

import fme
from fme.ace.train.train_config import TrainConfig
from fme.core.cli import prepare_config
from fme.core.distributed import Distributed
from fme.core.optimization import NullOptimization


def _load_train_config(
    config_path: str, experiment_dir: Path, batch_size: int | None
) -> TrainConfig:
    config_data = prepare_config(config_path)
    if config_data.get("experiment_dir") in ("FME_OUTPUT_DIR", None):
        config_data["experiment_dir"] = str(experiment_dir)
    if batch_size is not None:
        config_data.setdefault("train_loader", {})["batch_size"] = batch_size
    config = dacite.from_dict(
        data_class=TrainConfig,
        data=config_data,
        config=dacite.Config(strict=True),
    )
    config.set_random_seed()
    return config


def _loss_weights(config: TrainConfig) -> dict[str, float]:
    loss_cfg = config.stepper_training.loss
    if loss_cfg is None or loss_cfg.weights is None:
        return {}
    return dict(loss_cfg.weights)


def _run_one_batch(
    config_path: str, label: str, batch_size: int | None
) -> tuple[float, list[tuple]]:
    with tempfile.TemporaryDirectory(prefix="ace-loss-diag-") as tmp:
        config = _load_train_config(config_path, Path(tmp), batch_size)
        weights = _loss_weights(config)

        with Distributed.context():
            logging.info("[%s] Building train loader", label)
            train_data = config._get_train_data()
            logging.info("[%s] Initializing stepper", label)
            stepper = config._get_stepper(dataset_info=train_data.dataset_info)
            stepper.set_train()

            batch = next(iter(train_data.loader))
            logging.info(
                "[%s] Running one train_on_batch on device=%s",
                label,
                fme.get_device(),
            )
            output = stepper.train_on_batch(
                batch,
                optimization=NullOptimization(),
            )
            if fme.using_gpu():
                torch.cuda.synchronize()

            total_loss = float(output.get_metrics()["loss"])
            if output.per_channel_losses is None:
                raise RuntimeError("per_channel_losses missing from TrainOutput")

            rows = []
            for name, info in output.per_channel_losses.items():
                loss_val = float(info.loss.detach().cpu())
                rows.append(
                    (
                        name,
                        loss_val,
                        weights.get(name, 1.0),
                        int(info.count),
                    )
                )
            rows.sort(key=lambda r: r[1], reverse=True)
            return total_loss, rows


def _print_report(
    label: str,
    config_path: str,
    total_loss: float,
    rows: list[tuple],
    top_n: int,
) -> None:
    n_channels = len(rows)
    mean_ch_loss = sum(r[1] for r in rows) / n_channels if n_channels else 0.0
    print()
    print(f"=== {label} ===")
    print(f"config: {config_path}")
    print(f"total batch loss (sum over forward steps): {total_loss:.6g}")
    print(f"channels: {n_channels}")
    print(f"mean per-channel loss: {mean_ch_loss:.6g}")
    print()
    print(
        f"{'rank':>4}  {'variable':<24}  {'loss':>12}  {'wgt':>6}  "
        f"{'loss/mean':>10}  {'% of sum':>8}"
    )
    print("-" * 72)
    loss_sum = sum(r[1] for r in rows) or 1.0
    for rank, (name, loss_val, weight, count) in enumerate(rows[:top_n], start=1):
        pct = 100.0 * loss_val / loss_sum
        ratio = loss_val / mean_ch_loss if mean_ch_loss > 0 else float("nan")
        print(
            f"{rank:4d}  {name:<24}  {loss_val:12.6g}  {weight:6.2g}  "
            f"{ratio:10.2f}  {pct:7.1f}%"
        )
    if n_channels > top_n:
        tail_loss = sum(r[1] for r in rows[top_n:])
        print(
            f"     ... {n_channels - top_n} more channels, "
            f"combined loss={tail_loss:.6g} "
            f"({100.0 * tail_loss / loss_sum:.1f}% of channel sum)"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="ACE train config yaml")
    parser.add_argument(
        "--compare",
        help="Optional second config (e.g. clim-forcing) for side-by-side context",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of highest-loss variables to print (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Train loader batch size for the diagnostic "
            "(default: 1; avoids OOM on 1 GPU)"
        ),
    )
    args = parser.parse_args()

    label = Path(args.config).stem
    total_loss, rows = _run_one_batch(args.config, label, args.batch_size)
    _print_report(label, args.config, total_loss, rows, args.top)

    if args.compare:
        compare_label = Path(args.compare).stem
        compare_total, compare_rows = _run_one_batch(
            args.compare, compare_label, args.batch_size
        )
        _print_report(
            compare_label, args.compare, compare_total, compare_rows, args.top
        )

        prog_by_name = {r[0]: r[1] for r in rows}
        force_by_name = {r[0]: r[1] for r in compare_rows}
        shared = sorted(set(prog_by_name) & set(force_by_name))
        if shared:
            print()
            print("=== shared output channels: prognostic / forcing loss ratio ===")
            ratios = [
                (name, prog_by_name[name], force_by_name[name]) for name in shared
            ]
            ratios.sort(key=lambda t: t[1] / max(t[2], 1e-30), reverse=True)
            print(f"{'variable':<24}  {'prog':>12}  {'forcing':>12}  {'ratio':>8}")
            print("-" * 62)
            for name, prog, force in ratios[: args.top]:
                ratio = prog / force if force > 0 else float("inf")
                print(f"{name:<24}  {prog:12.6g}  {force:12.6g}  {ratio:8.1f}x")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Per-channel loss diagnostic failed")
        sys.exit(1)
