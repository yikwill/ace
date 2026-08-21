#!/usr/bin/env python3
"""Verify an ACE train config can load zarr data and run one train step."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Use the checkout that contains this script, not a sibling worktree.
_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))
os.environ.setdefault("PYTHONPATH", str(_WORKTREE))

import dacite
import torch

import fme
from fme.ace.train.train_config import TrainConfig
from fme.core.cli import prepare_config
from fme.core.distributed import Distributed


def main(config_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config_data = prepare_config(config_path)
    config = dacite.from_dict(
        data_class=TrainConfig,
        data=config_data,
        config=dacite.Config(strict=True),
    )
    config.set_random_seed()

    with Distributed.context():
        logging.info("Building train loader from zarr/netcdf config")
        train_data = config._get_train_data()
        train_data.log_info("train")

        logging.info("Initializing stepper")
        stepper = config._get_stepper(dataset_info=train_data.dataset_info)
        optimization = config._get_optimization(stepper.modules)

        logging.info("Fetching one training batch")
        batch = next(iter(train_data.loader))
        logging.info("Running one train step on device=%s", fme.get_device())
        output = stepper.train_on_batch(batch, optimization=optimization)
        metrics = output.get_metrics()
        logging.info(
            "Train step OK. sample metrics: %s",
            {k: float(v) for k, v in sorted(metrics.items())[:5]},
        )

        logging.info("Fetching one validation batch")
        validation_entries = config._get_validation_data()
        _, val_data, val_name = validation_entries[0]
        val_data.log_info(val_name)
        next(iter(val_data.loader))
        logging.info("Validation batch load OK")

        if fme.using_gpu():
            torch.cuda.synchronize()
        logging.info("Zarr training verification passed for %s", config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to ACE train config yaml")
    args = parser.parse_args()
    try:
        main(args.config)
    except Exception:
        logging.exception("Zarr training verification failed")
        sys.exit(1)
