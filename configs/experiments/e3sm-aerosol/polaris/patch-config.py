#!/usr/bin/env python3
"""Patch staged train configs for Polaris paths and optional debug overrides."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

PATH_REWRITES = [
    (
        "/p/lustre5/yik1/datasets",
        os.environ.get("FME_DATA_ROOT", "/eagle/E3SMinput/yikwill/datasets"),
    ),
    (
        "/global/cfs/projectdirs/e3sm/yikwill/datasets",
        os.environ.get("FME_DATA_ROOT", "/eagle/E3SMinput/yikwill/datasets"),
    ),
    (
        "/pscratch/sd/y/yikwill",
        os.environ.get("FME_DATA_ROOT", "/eagle/E3SMinput/yikwill/datasets"),
    ),
]


def rewrite_paths(obj):
    if isinstance(obj, dict):
        return {k: rewrite_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite_paths(v) for v in obj]
    if isinstance(obj, str):
        out = obj
        for old, new in PATH_REWRITES:
            out = out.replace(old, new)
        return out
    return obj


def apply_debug_patches(cfg: dict, max_epochs: int) -> dict:
    cfg["max_epochs"] = max_epochs
    logging = cfg.setdefault("logging", {})
    logging["log_to_wandb"] = False
    logging["log_to_file"] = False
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--debug-max-epochs", type=int, default=None)
    args = parser.parse_args()

    with open(args.input) as f:
        cfg = yaml.safe_load(f)

    cfg = rewrite_paths(cfg)
    if args.debug_max_epochs is not None:
        cfg = apply_debug_patches(cfg, args.debug_max_epochs)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("patch-config failed", file=sys.stderr)
        raise
