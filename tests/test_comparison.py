import copy

import pytest

from src.comparison import ComparisonManifestError, validate_comparison_metadata


def comparison_inputs():
    manifest = {
        "seed": 19,
        "split_files": {
            "train": {"sha256": "train", "rows": 11},
            "validation": {"sha256": "val", "rows": 3},
        },
    }
    config = {
        "model_id": "distilbert-base-uncased",
        "model_revision": "a" * 40,
        "tokenizer_id": "distilbert-base-uncased",
        "tokenizer_revision": "b" * 40,
        "seed": 19,
        "max_expanded_length": 128,
        "max_byte_span_length": 512,
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "epochs": 3,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "gradient_clip_norm": 1.0,
        "mixed_precision": "fp16",
        "validation_interval_steps": 1000,
        "num_workers": 0,
        "device": "cuda",
        "dropout": 0.1,
        "byte_num_filters": 48,
    }
    execution = {
        "resolved_device": "cuda:0",
        "effective_precision": "float16",
        "cuda_available": True,
        "cuda_device_count": 1,
        "cuda_device_name": "fixture GPU",
        "cuda_capability": [8, 0],
        "cuda_bf16_supported": True,
    }
    common = {
        "training_config": config,
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "tokenizer_id": config["tokenizer_id"],
        "tokenizer_revision": config["tokenizer_revision"],
        "execution": execution,
        "dataset_provenance": manifest,
        "label_mapping": {"0": "Negative", "1": "Positive"},
    }
    baseline = {**common, "model_kind": "baseline", "architecture": {"pooling": "cls"}}
    canonical = {**copy.deepcopy(common), "model_kind": "canonical", "architecture": {"pooling": "mean", "byte_width": 64}}
    canonical["training_config"]["dropout"] = 0.2
    canonical["training_config"]["byte_num_filters"] = 96
    return baseline, canonical, manifest


def test_comparison_manifest_accepts_shared_controls_and_model_specific_architecture():
    baseline, canonical, manifest = comparison_inputs()

    report = validate_comparison_metadata(baseline, canonical, manifest)

    assert report["comparison_type"] == "whole-system; byte-feature effect is not isolated"
    assert report["shared_settings"]["effective_batch_size"] == 16
    assert report["shared_settings"]["optimizer"] == "AdamW"
    assert report["shared_settings"]["checkpoint_selection"] == "minimum validation loss; strict improvement only"
    assert report["shared_settings"]["stopping_rule"] == "fixed epoch count; no early stopping"
    assert report["shared_settings"]["scheduler"] == "linear-warmup-decay"
    assert report["shared_settings"]["optimizer_settings"] == {
        "betas": [0.9, 0.999], "epsilon": 1e-8, "weight_decay": 0.01, "amsgrad": False
    }
    assert report["shared_settings"]["scheduler_settings"]["total_optimizer_steps"] == 3
    assert report["shared_settings"]["scheduler_settings"]["warmup_steps"] == 0


@pytest.mark.parametrize("field", ["seed", "mixed_precision", "learning_rate", "gradient_accumulation_steps"])
def test_comparison_manifest_rejects_shared_setting_mismatch(field):
    baseline, canonical, manifest = comparison_inputs()
    canonical["training_config"][field] = "different" if isinstance(canonical["training_config"][field], str) else -1

    with pytest.raises(ComparisonManifestError, match=field):
        validate_comparison_metadata(baseline, canonical, manifest)


def test_comparison_manifest_rejects_dataset_or_execution_mismatch():
    baseline, canonical, manifest = comparison_inputs()
    canonical["dataset_provenance"] = {"seed": 20}
    canonical["execution"] = {**canonical["execution"], "effective_precision": "bfloat16"}

    with pytest.raises(ComparisonManifestError, match="execution.effective_precision|dataset provenance"):
        validate_comparison_metadata(baseline, canonical, manifest)
