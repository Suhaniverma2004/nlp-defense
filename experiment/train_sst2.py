"""Train the baseline or canonical byte-aware SST-2 classifier."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from src.sst2_data import load_split
from src.training import (
    PINNED_REPRODUCIBILITY_REVISION,
    TrainingConfig,
    fit_model,
    load_pretrained_components,
    save_training_history,
)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=("baseline", "canonical"), required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="distilbert-base-uncased")
    parser.add_argument("--model-revision", default=PINNED_REPRODUCIBILITY_REVISION)
    parser.add_argument("--tokenizer-id", default="distilbert-base-uncased")
    parser.add_argument("--tokenizer-revision", default=PINNED_REPRODUCIBILITY_REVISION)
    parser.add_argument("--max-expanded-length", type=int, default=128)
    parser.add_argument("--max-byte-span-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--mixed-precision", choices=("none", "fp16", "bf16"), default="none")
    parser.add_argument("--validation-interval-steps", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume-from", type=Path, help="resume from a trusted epoch-boundary checkpoint bundle")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--byte-embedding-dim", type=int, default=32)
    parser.add_argument("--byte-num-filters", type=int, default=48)
    parser.add_argument("--byte-kernel-sizes", type=int, nargs="+", default=(2, 3, 4))
    parser.add_argument("--byte-output-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--allow-hub-download", action="store_true", help="explicitly allow loading the pinned revisions from the Hub if absent locally")
    args = parser.parse_args()

    train_frame, manifest = load_split(args.split_dir, "train")
    validation_frame, validation_manifest = load_split(args.split_dir, "validation")
    if manifest != validation_manifest:
        raise ValueError("train and validation files do not share one split manifest.")
    config = TrainingConfig(
        model_id=args.model_id,
        model_revision=args.model_revision,
        tokenizer_id=args.tokenizer_id,
        tokenizer_revision=args.tokenizer_revision,
        max_expanded_length=args.max_expanded_length,
        max_byte_span_length=args.max_byte_span_length,
        batch_size=args.batch_size,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_clip_norm=args.gradient_clip_norm,
        mixed_precision=args.mixed_precision,
        validation_interval_steps=args.validation_interval_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        device=args.device,
        output_dir=str(args.output_dir),
        byte_embedding_dim=args.byte_embedding_dim,
        byte_num_filters=args.byte_num_filters,
        byte_kernel_sizes=tuple(args.byte_kernel_sizes),
        byte_output_dim=args.byte_output_dim,
        dropout=args.dropout,
    )
    if args.output_dir.exists():
        existing = {item.name for item in args.output_dir.iterdir()}
        if args.resume_from is None and existing:
            raise FileExistsError(f"run output directory is not empty: {args.output_dir}")
        if args.resume_from is not None:
            allowed = {
                "checkpoints",
                "experiment_config.json",
                "training_history.json",
                "training_history.csv",
                "best_checkpoint.json",
            }
            known_temporary_prefixes = (
                ".training_history.json.tmp-",
                ".training_history.csv.tmp-",
                ".best_checkpoint.json.tmp-",
                ".experiment_config.json.tmp-",
            )
            unexpected = {
                name for name in existing
                if name not in allowed and not name.startswith(known_temporary_prefixes)
            }
            if unexpected:
                raise FileExistsError(
                    f"resume output directory contains unrelated entries: {sorted(unexpected)}"
                )

    config_path = args.output_dir / "experiment_config.json"
    serialized_config = json.loads(json.dumps(asdict(config)))
    if args.resume_from is not None and config_path.is_file():
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        previous_config.pop("output_dir", None)
        comparison_config = dict(serialized_config)
        comparison_config.pop("output_dir", None)
        if previous_config != comparison_config:
            raise ValueError("resume arguments differ from the saved experiment configuration.")
    elif config_path.exists() and args.resume_from is None:
        raise FileExistsError(f"experiment configuration already exists: {config_path}")

    model, tokenizer = load_pretrained_components(
        config,
        model_kind=args.model_kind,
        local_files_only=not args.allow_hub_download,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        _atomic_write_text(config_path, json.dumps(asdict(config), indent=2, sort_keys=True) + "\n")
    result = fit_model(
        model,
        tokenizer,
        train_frame,
        validation_frame,
        config,
        model_kind=args.model_kind,
        checkpoint_root=args.output_dir / "checkpoints",
        dataset_provenance=manifest,
        resume_checkpoint_dir=args.resume_from,
        history_path=args.output_dir / "training_history.json",
    )
    save_training_history(result.history, args.output_dir / "training_history.json")
    if result.best_checkpoint_dir is None:
        raise RuntimeError("training did not create a validation-selected checkpoint.")
    pointer = args.output_dir / "best_checkpoint.json"
    _atomic_write_text(
        pointer,
        json.dumps(
            {
                "checkpoint_dir": str(Path(result.best_checkpoint_dir).relative_to(args.output_dir)),
                "selection_metric": "validation loss (minimum)",
                "best_validation_metrics": result.best_validation_metrics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(json.dumps({"device": result.device, "mixed_precision": result.effective_mixed_precision, "best_checkpoint": result.best_checkpoint_dir, "best_validation_metrics": result.best_validation_metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
