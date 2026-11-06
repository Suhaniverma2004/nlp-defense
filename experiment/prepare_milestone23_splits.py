"""Inspect local SST-2 files and create Milestone 2.3 splits without downloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.sst2_data import inspect_sst2_sources, prepare_reproducible_splits

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data/sst2/sst2_train.csv")
    parser.add_argument("--original-validation", type=Path, default=ROOT / "data/sst2/sst2_validation.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/milestone2.3-seed42")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(inspect_sst2_sources(args.train, args.original_validation), indent=2, sort_keys=True))
    manifest = prepare_reproducible_splits(
        args.train,
        args.original_validation,
        args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
