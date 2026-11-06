"""Validate shared controls before comparing baseline and canonical runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.comparison import validate_comparison_checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--canonical-checkpoint", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    args = parser.parse_args()
    report = validate_comparison_checkpoints(
        args.baseline_checkpoint,
        args.canonical_checkpoint,
        args.split_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
