"""Evaluate a selected checkpoint on manifest-verified validation or test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation import evaluate_checkpoint_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = evaluate_checkpoint_split(
        args.checkpoint_dir,
        args.split_dir,
        args.split,
        args.output_dir,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
