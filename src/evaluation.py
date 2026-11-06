"""Leakage-aware clean validation and held-out test evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.checkpointing import load_checkpoint
from src.sst2_data import load_split
from src.training import (
    TrainingConfig,
    _build_loader,
    evaluate_loader,
    resolve_device,
    resolve_mixed_precision,
)


def evaluate_checkpoint_split(
    checkpoint_dir: str | Path,
    split_dir: str | Path,
    split_name: str,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Evaluate a selected checkpoint on one manifest-verified split.

    This interface does not train or select a checkpoint. In particular, the
    held-out test split is only read here, after the caller has selected a
    checkpoint using validation metrics.
    """
    if split_name not in {"validation", "test"}:
        raise ValueError("evaluation split must be validation or test.")
    frame, manifest = load_split(split_dir, split_name)
    model, tokenizer, metadata = load_checkpoint(checkpoint_dir)
    saved_manifest = metadata.get("dataset_provenance")
    if not isinstance(saved_manifest, dict):
        raise ValueError("checkpoint has no dataset split provenance; refusing unverifiable evaluation.")
    saved_entry = saved_manifest.get("split_files", {}).get(split_name)
    current_entry = manifest.get("split_files", {}).get(split_name)
    if not saved_entry or saved_entry != current_entry:
        raise ValueError(f"{split_name} split hash does not match the selected checkpoint's recorded dataset.")
    if saved_manifest.get("source", {}).get("train_sha256") != manifest.get("source", {}).get("train_sha256"):
        raise ValueError("source training-file hash differs from checkpoint split provenance.")

    config = TrainingConfig(**metadata["training_config"])
    device = resolve_device(device_name)
    model = model.to(device)
    amp_dtype = resolve_mixed_precision(config.mixed_precision, device)
    loader = _build_loader(frame, tokenizer, config, metadata["model_kind"], shuffle=False)
    metrics = evaluate_loader(model, loader, device=device, autocast_dtype=amp_dtype)
    output = {
        "split": split_name,
        "split_sha256": current_entry["sha256"],
        "source_train_sha256": manifest["source"]["train_sha256"],
        "split_seed": manifest["seed"],
        "model_kind": metadata["model_kind"],
        "model_id": metadata["model_id"],
        "model_revision": metadata["model_revision"],
        "tokenizer_id": metadata["tokenizer_id"],
        "tokenizer_revision": metadata["tokenizer_revision"],
        "device": str(device),
        "metrics": metrics,
        "interpretation": (
            "Clean SST-2 sentiment classification metrics; these do not measure adversarial attack detection or robustness."
        ),
    }
    output_dir = Path(output_dir)
    stem = f"{split_name}_metrics"
    output_paths = (output_dir / f"{stem}.json", output_dir / f"{stem}.csv")
    if any(path.exists() for path in output_paths):
        raise FileExistsError(f"evaluation output already exists in {output_dir}; choose a new output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{stem}.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pd.json_normalize(output, sep=".").to_csv(output_dir / f"{stem}.csv", index=False)
    return output
