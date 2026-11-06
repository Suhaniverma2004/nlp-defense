"""Validation of shared settings before whole-system run comparisons."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from src.checkpointing import validate_checkpoint_bundle
from src.sst2_data import load_split

SHARED_CONFIG_FIELDS = (
    "model_id",
    "model_revision",
    "tokenizer_id",
    "tokenizer_revision",
    "seed",
    "max_expanded_length",
    "max_byte_span_length",
    "batch_size",
    "gradient_accumulation_steps",
    "epochs",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "gradient_clip_norm",
    "mixed_precision",
    "validation_interval_steps",
    "num_workers",
    "device",
)


class ComparisonManifestError(ValueError):
    """Raised when two checkpoints do not share a controlled run configuration."""


def validate_comparison_metadata(
    baseline: dict[str, Any],
    canonical: dict[str, Any],
    split_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Check shared experimental controls while allowing model-specific architecture."""
    differences: list[str] = []
    baseline_config = baseline.get("training_config")
    canonical_config = canonical.get("training_config")
    if not isinstance(baseline_config, dict) or not isinstance(canonical_config, dict):
        raise ComparisonManifestError("both checkpoints must contain a training_config object.")

    shared: dict[str, Any] = {}
    for field in SHARED_CONFIG_FIELDS:
        left, right = baseline_config.get(field), canonical_config.get(field)
        if left != right:
            differences.append(f"{field}: baseline={left!r}, canonical={right!r}")
        else:
            shared[field] = left

    for field in ("batch_size", "gradient_accumulation_steps"):
        value = baseline_config.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            differences.append(f"{field} must be a positive integer in both run manifests")
    for field in ("epochs",):
        value = baseline_config.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            differences.append(f"{field} must be a positive integer in both run manifests")

    batch_size = baseline_config.get("batch_size")
    accumulation_steps = baseline_config.get("gradient_accumulation_steps")
    if (
        isinstance(batch_size, int)
        and not isinstance(batch_size, bool)
        and batch_size > 0
        and isinstance(accumulation_steps, int)
        and not isinstance(accumulation_steps, bool)
        and accumulation_steps > 0
    ):
        shared["effective_batch_size"] = batch_size * accumulation_steps
    shared["optimizer"] = "AdamW"
    shared["checkpoint_selection"] = "minimum validation loss; strict improvement only"
    shared["stopping_rule"] = "fixed epoch count; no early stopping"
    shared["optimizer_settings"] = {
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": baseline_config.get("weight_decay"),
        "amsgrad": False,
    }
    shared["scheduler"] = "linear-warmup-decay"
    train_rows = split_manifest.get("split_files", {}).get("train", {}).get("rows")
    valid_schedule_inputs = (
        isinstance(train_rows, int)
        and not isinstance(train_rows, bool)
        and train_rows > 0
        and isinstance(batch_size, int)
        and not isinstance(batch_size, bool)
        and batch_size > 0
        and isinstance(accumulation_steps, int)
        and not isinstance(accumulation_steps, bool)
        and accumulation_steps > 0
        and isinstance(baseline_config.get("epochs"), int)
        and not isinstance(baseline_config.get("epochs"), bool)
        and baseline_config["epochs"] > 0
        and isinstance(baseline_config.get("warmup_ratio"), (int, float))
        and math.isfinite(baseline_config["warmup_ratio"])
    )
    if valid_schedule_inputs:
        microbatches = math.ceil(train_rows / baseline_config["batch_size"])
        updates_per_epoch = math.ceil(microbatches / baseline_config["gradient_accumulation_steps"])
        total_updates = updates_per_epoch * baseline_config["epochs"]
        warmup_steps = int(total_updates * baseline_config["warmup_ratio"])
    else:
        if not isinstance(train_rows, int) or isinstance(train_rows, bool) or train_rows <= 0:
            differences.append("split manifest must include a positive train row count")
        total_updates = warmup_steps = None
    shared["scheduler_settings"] = {
        "name": "LambdaLR linear warmup/decay",
        "warmup_ratio": baseline_config.get("warmup_ratio"),
        "warmup_steps": warmup_steps,
        "total_optimizer_steps": total_updates,
    }

    for field in ("model_id", "model_revision", "tokenizer_id", "tokenizer_revision"):
        left, right = baseline.get(field), canonical.get(field)
        if left != right:
            differences.append(f"checkpoint {field}: baseline={left!r}, canonical={right!r}")

    left_execution = baseline.get("execution")
    right_execution = canonical.get("execution")
    if not isinstance(left_execution, dict) or not isinstance(right_execution, dict):
        differences.append("execution metadata is missing from one or both checkpoints")
    else:
        for field in (
            "resolved_device",
            "effective_precision",
            "cuda_available",
            "cuda_device_count",
            "cuda_device_name",
            "cuda_capability",
            "cuda_bf16_supported",
        ):
            if left_execution.get(field) != right_execution.get(field):
                differences.append(
                    f"execution.{field}: baseline={left_execution.get(field)!r}, "
                    f"canonical={right_execution.get(field)!r}"
                )

    if baseline.get("dataset_provenance") != split_manifest:
        differences.append("baseline dataset provenance does not match the supplied split manifest")
    if canonical.get("dataset_provenance") != split_manifest:
        differences.append("canonical dataset provenance does not match the supplied split manifest")
    if baseline.get("dataset_provenance") != canonical.get("dataset_provenance"):
        differences.append("dataset provenance differs between checkpoints")
    if baseline.get("label_mapping") != canonical.get("label_mapping"):
        differences.append("label mappings differ between checkpoints")

    if differences:
        raise ComparisonManifestError("comparison controls do not match:\n- " + "\n- ".join(differences))
    return {
        "comparison_type": "whole-system; byte-feature effect is not isolated",
        "model_kinds": [baseline.get("model_kind"), canonical.get("model_kind")],
        "shared_settings": shared,
        "dataset_manifest": split_manifest,
        "label_mapping": baseline.get("label_mapping"),
    }


def validate_comparison_checkpoints(
    baseline_checkpoint: str | Path,
    canonical_checkpoint: str | Path,
    split_dir: str | Path,
) -> dict[str, Any]:
    """Validate checkpoint bundles and shared settings against a local split manifest."""
    baseline = validate_checkpoint_bundle(baseline_checkpoint)
    canonical = validate_checkpoint_bundle(canonical_checkpoint)
    _frame, split_manifest = load_split(split_dir, "train")
    return validate_comparison_metadata(baseline, canonical, split_manifest)
