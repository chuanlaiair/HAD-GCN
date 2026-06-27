"""Command-line entry point for HAD-GCN."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .config import (
        DATA_FOLDER,
        SUBJECT_ID,
        ExperimentConfig,
    )
    from .experiment import run_experiment
    from .utils import configure_logging
except ImportError:
    from config import (
        DATA_FOLDER,
        SUBJECT_ID,
        ExperimentConfig,
    )
    from experiment import run_experiment
    from utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the public HAD-GCN reproduction."
    )
    parser.add_argument(
        "--data-folder",
        type=Path,
        default=DATA_FOLDER,
    )
    parser.add_argument(
        "--subject-id",
        type=int,
        default=SUBJECT_ID,
        choices=range(1, 10),
    )
    parser.add_argument(
        "--protocol",
        choices=("subject_specific", "cross_subject"),
        default="subject_specific",
    )
    parser.add_argument(
        "--branch",
        choices=("raw", "cwt"),
        default="raw",
        help=(
            "Manual patent placeholder. This is not the adaptive selector."
        ),
    )
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--disable-calibration",
        action="store_true",
    )
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
    )
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()

    config = ExperimentConfig(
        data_folder=args.data_folder,
        subject_id=args.subject_id,
        protocol=args.protocol,
        manual_branch_placeholder=args.branch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        calibration_enabled=not args.disable_calibration,
        early_stopping_enabled=not args.disable_early_stopping,
        rebuild_cache=args.rebuild_cache,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
