import hashlib
import json
import random
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import load_file, save_file

from src.checkpointing import (
    CheckpointError,
    load_checkpoint,
    load_training_state,
    restore_rng_state,
    save_checkpoint,
    save_training_state,
    recover_latest_resume_checkpoint,
    write_latest_resume_pointer,
)
from src.canonical_model import CanonicalByteAwareDistilBert
from src.evaluation import evaluate_checkpoint_split
from src.sst2_data import prepare_reproducible_splits
from src.training import (
    ClassificationCollator,
    DistilBertBaselineClassifier,
    PINNED_REPRODUCIBILITY_REVISION,
    TrainingConfig,
)
from tests.milestone23_fixtures import example_frame, local_fast_tokenizer, tiny_backbone


def make_config():
    return TrainingConfig(
        model_revision=PINNED_REPRODUCIBILITY_REVISION,
        tokenizer_revision=PINNED_REPRODUCIBILITY_REVISION,
        max_expanded_length=32,
        max_byte_span_length=64,
        batch_size=2,
        epochs=1,
        warmup_ratio=0.0,
        device="cpu",
        byte_embedding_dim=8,
        byte_num_filters=4,
        byte_kernel_sizes=(2, 3),
        byte_output_dim=8,
    )


def make_model(model_kind, tokenizer):
    backbone = tiny_backbone(len(tokenizer), seed=37)
    if model_kind == "baseline":
        return DistilBertBaselineClassifier(backbone, dropout=0.0)
    return CanonicalByteAwareDistilBert(
        backbone,
        byte_embedding_dim=8,
        byte_num_filters=4,
        byte_kernel_sizes=(2, 3),
        byte_output_dim=8,
        dropout=0.0,
        pad_token_id=tokenizer.pad_token_id,
    )


def forward(model, tokenizer, kind):
    batch = ClassificationCollator(tokenizer, make_config(), kind)([("good movie", 1, 0)])
    if kind == "canonical":
        return model(batch["inputs"]).logits
    return model(**batch["inputs"]).logits


@pytest.mark.parametrize("model_kind", ["baseline", "canonical"])
def test_versioned_checkpoint_round_trip_is_offline_and_strict(model_kind, tmp_path, monkeypatch):
    tokenizer = local_fast_tokenizer()
    model = make_model(model_kind, tokenizer).eval()
    expected_logits = forward(model, tokenizer, model_kind)
    config = make_config()
    provenance = {
        "source": {"train_sha256": "train-hash"},
        "split_files": {
            "train": {"path": "train.csv", "sha256": "train-split-hash"},
            "validation": {"path": "validation.csv", "sha256": "validation-hash"},
            "test": {"path": "test.csv", "sha256": "test-hash"},
        },
    }
    checkpoint = tmp_path / model_kind
    save_checkpoint(
        model,
        tokenizer,
        checkpoint,
        model_kind=model_kind,
        training_config=config.__dict__,
        dataset_provenance=provenance,
        best_validation_metrics={"loss": 0.5, "accuracy": 0.75},
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    loaded, loaded_tokenizer, metadata = load_checkpoint(
        checkpoint,
        expected_model_kind=model_kind,
        expected_model_id=config.model_id,
        expected_model_revision=config.model_revision,
        expected_tokenizer_id=config.tokenizer_id,
        expected_tokenizer_revision=config.tokenizer_revision,
    )
    actual_logits = forward(loaded, loaded_tokenizer, model_kind)

    assert loaded_tokenizer.is_fast
    assert metadata["schema_version"] == 1
    assert (checkpoint / "model.safetensors").is_file()
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)

    with pytest.raises(CheckpointError, match="does not match expected"):
        load_checkpoint(checkpoint, expected_tokenizer_revision="0" * 40)


def test_checkpoint_strictly_rejects_missing_weight_key_even_with_updated_hash(tmp_path):
    tokenizer = local_fast_tokenizer()
    model = make_model("baseline", tokenizer)
    checkpoint = tmp_path / "checkpoint"
    config = make_config()
    save_checkpoint(
        model,
        tokenizer,
        checkpoint,
        model_kind="baseline",
        training_config=config.__dict__,
        dataset_provenance={},
        best_validation_metrics={"loss": 1.0},
    )
    weights_path = checkpoint / "model.safetensors"
    state = load_file(str(weights_path))
    malformed_checkpoint = tmp_path / "malformed-checkpoint"
    malformed_checkpoint.mkdir()
    shutil.copytree(checkpoint / "tokenizer", malformed_checkpoint / "tokenizer")
    malformed_path = malformed_checkpoint / "model.safetensors"
    save_file({key: value.clone() for key, value in state.items() if key != "classifier.weight"}, str(malformed_path))
    metadata_path = malformed_checkpoint / "metadata.json"
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    metadata["weights_sha256"] = hashlib.sha256(malformed_path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(CheckpointError, match="strict loading"):
        load_checkpoint(malformed_checkpoint)


def test_training_state_is_separate_and_requires_explicit_trust(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "trusted-training-state.pt"
    save_training_state(path, optimizer=optimizer, scheduler=scheduler, epoch=2, global_step=9)

    with pytest.raises(CheckpointError, match="trusted=True"):
        load_training_state(path)
    state = load_training_state(path, trusted=True)
    assert state["epoch"] == 2
    assert state["global_step"] == 9
    assert state["training_state_version"] == 2
    assert {"python", "numpy", "torch_cpu", "torch_cuda", "data_loader_generator"} <= set(state["rng"])


def test_training_state_restores_python_numpy_torch_and_loader_rng(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    generator = torch.Generator().manual_seed(707)
    random.seed(71)
    np.random.seed(72)
    torch.manual_seed(73)
    path = tmp_path / "rng-state.pt"

    save_training_state(
        path,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        global_step=4,
        data_loader_generator=generator,
    )
    expected = (random.random(), np.random.random(), torch.rand(1), torch.rand(1, generator=generator))

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    generator.manual_seed(999)
    state = load_training_state(path, trusted=True)
    restore_rng_state(state, data_loader_generator=generator)
    actual = (random.random(), np.random.random(), torch.rand(1), torch.rand(1, generator=generator))

    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)


def test_atomic_history_publication_keeps_previous_json_if_commit_replace_fails(tmp_path, monkeypatch):
    import src.training as training_module

    history = tmp_path / "training_history.json"
    history.write_text('[{"event":"previous"}]\n', encoding="utf-8")
    original_replace = training_module.os.replace

    def fail_json_commit(source, destination):
        if str(destination) == str(history):
            raise OSError("simulated interruption before JSON history commit")
        return original_replace(source, destination)

    monkeypatch.setattr(training_module.os, "replace", fail_json_commit)
    with pytest.raises(OSError, match="simulated interruption"):
        training_module.save_training_history([{"event": "new"}], history)

    assert history.read_text(encoding="utf-8") == '[{"event":"previous"}]\n'


def test_latest_pointer_is_atomically_repaired_from_newest_valid_snapshot(tmp_path, monkeypatch):
    # A malformed/stale pointer is never trusted over validated immutable bundles.
    root = tmp_path / "checkpoints"
    root.mkdir()
    old_pointer = root / "latest_checkpoint.json"
    old_pointer.write_text("{broken", encoding="utf-8")
    # No valid snapshots means recovery does not manufacture a target or bless the pointer.
    assert recover_latest_resume_checkpoint(root) is None
    assert old_pointer.read_text(encoding="utf-8") == "{broken"


def test_interrupted_latest_pointer_replacement_keeps_previous_pointer(tmp_path, monkeypatch):
    import src.checkpointing as checkpointing_module

    root = tmp_path / "checkpoints"
    root.mkdir()
    pointer = root / "latest_checkpoint.json"
    pointer.write_text('{"checkpoint_dir":"previous"}\n', encoding="utf-8")
    original_replace = checkpointing_module.os.replace

    def fail_pointer_replace(source, destination):
        if str(destination) == str(pointer):
            raise OSError("simulated interruption during pointer publication")
        return original_replace(source, destination)

    monkeypatch.setattr(checkpointing_module.os, "replace", fail_pointer_replace)
    with pytest.raises(OSError, match="pointer publication"):
        write_latest_resume_pointer(root, root / "next", 2, 10)
    assert pointer.read_text(encoding="utf-8") == '{"checkpoint_dir":"previous"}\n'


def test_incomplete_abandoned_resume_directory_is_ignored_and_preserved(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    abandoned = root / ".resume-epoch-0001-step-00000001-deadbeef.tmp-incomplete"
    abandoned.mkdir()
    (abandoned / "metadata.json").write_text("{}", encoding="utf-8")

    with pytest.warns(RuntimeWarning, match="Ignoring incomplete or invalid staging"):
        assert recover_latest_resume_checkpoint(root) is None
    assert abandoned.is_dir()


def test_evaluation_moves_loaded_model_to_selected_device(monkeypatch, tmp_path):
    import src.evaluation as evaluation_module

    config = make_config()
    model = torch.nn.Linear(2, 2)
    tokenizer = local_fast_tokenizer()
    manifest = {
        "source": {"train_sha256": "source-hash"},
        "split_files": {"validation": {"path": "validation.csv", "sha256": "validation-hash"}},
        "seed": 42,
    }
    monkeypatch.setattr(evaluation_module, "load_split", lambda *_: (example_frame(), manifest))
    monkeypatch.setattr(
        evaluation_module,
        "load_checkpoint",
        lambda *_: (model, tokenizer, {
            "dataset_provenance": manifest,
            "training_config": config.__dict__,
            "model_kind": "baseline",
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "tokenizer_id": config.tokenizer_id,
            "tokenizer_revision": config.tokenizer_revision,
        }),
    )
    target_device = torch.device("meta")
    monkeypatch.setattr(evaluation_module, "resolve_device", lambda _: target_device)
    monkeypatch.setattr(evaluation_module, "_build_loader", lambda *_args, **_kwargs: object())

    def assert_placement(received_model, _loader, *, device, autocast_dtype):
        assert next(received_model.parameters()).device == target_device
        assert device == target_device
        return {"count": 0}

    monkeypatch.setattr(evaluation_module, "evaluate_loader", assert_placement)
    result = evaluation_module.evaluate_checkpoint_split(
        "checkpoint", "splits", "validation", tmp_path / "metrics", device_name="requested"
    )

    assert result["device"] == "meta"


def test_evaluation_writes_metrics_only_for_manifest_verified_test_split(tmp_path, monkeypatch):
    train_rows = []
    for label, name in ((0, "Negative"), (1, "Positive")):
        for index in range(20):
            train_rows.append(
                {
                    "sentence": f"{name.lower()} sample {index}",
                    "label": label,
                    "idx": len(train_rows),
                    "label_name": name,
                }
            )
    # Tokenizer unknowns are valid fixture tokens with exact source offsets.
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "original-validation.csv"
    pd.DataFrame(train_rows).to_csv(train_path, index=False)
    pd.DataFrame(
        [{"sentence": "official dev", "label": 0, "idx": 0, "label_name": "Negative"}]
    ).to_csv(validation_path, index=False)
    split_dir = tmp_path / "splits"
    manifest = prepare_reproducible_splits(train_path, validation_path, split_dir, seed=42)

    tokenizer = local_fast_tokenizer()
    model = make_model("baseline", tokenizer).eval()
    config = make_config()
    checkpoint = tmp_path / "selected"
    save_checkpoint(
        model,
        tokenizer,
        checkpoint,
        model_kind="baseline",
        training_config=config.__dict__,
        dataset_provenance=manifest,
        best_validation_metrics={"loss": 1.0},
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    output_dir = tmp_path / "evaluation"

    result = evaluate_checkpoint_split(checkpoint, split_dir, "test", output_dir, device_name="cpu")

    assert result["split"] == "test"
    assert result["split_sha256"] == manifest["split_files"]["test"]["sha256"]
    assert result["metrics"]["count"] == manifest["split_files"]["test"]["rows"]
    assert (output_dir / "test_metrics.json").is_file()
    assert (output_dir / "test_metrics.csv").is_file()
    assert "do not measure adversarial" in result["interpretation"]
