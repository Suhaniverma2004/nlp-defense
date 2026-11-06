import pickle
import shutil
import warnings
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import AutoTokenizer, DistilBertModel

from src.canonical_model import CanonicalByteAwareDistilBert
from src.alignment import align_text
from src.canonical_model import collate_alignment_results
from src.training import (
    ClassificationCollator,
    DistilBertBaselineClassifier,
    PINNED_REPRODUCIBILITY_REVISION,
    TrainingConfig,
    _build_loader,
    evaluate_loader,
    fit_model,
    load_pretrained_components,
    resolve_mixed_precision,
    validate_resume_execution,
    validate_labeled_frame,
)
from tests.milestone23_fixtures import example_frame, local_fast_tokenizer, tiny_backbone


def config(**overrides):
    values = {
        "model_revision": PINNED_REPRODUCIBILITY_REVISION,
        "tokenizer_revision": PINNED_REPRODUCIBILITY_REVISION,
        "max_expanded_length": 32,
        "max_byte_span_length": 64,
        "batch_size": 2,
        "seed": 17,
        "epochs": 1,
        "learning_rate": 1e-3,
        "warmup_ratio": 0.0,
        "mixed_precision": "none",
        "device": "cpu",
        "byte_embedding_dim": 8,
        "byte_num_filters": 4,
        "byte_kernel_sizes": (2, 3),
        "byte_output_dim": 8,
    }
    values.update(overrides)
    return TrainingConfig(**values)


def make_model(kind, tokenizer):
    backbone = tiny_backbone(len(tokenizer))
    if kind == "baseline":
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


def validation_frame():
    import pandas as pd

    return pd.DataFrame(
        [
            {"sentence": "good film", "label": 1, "idx": 10, "label_name": "Positive"},
            {"sentence": "bad movie", "label": 0, "idx": 11, "label_name": "Negative"},
            {"sentence": "great acting", "label": 1, "idx": 12, "label_name": "Positive"},
            {"sentence": "really boring", "label": 0, "idx": 13, "label_name": "Negative"},
        ]
    )


@pytest.mark.parametrize("model_kind", ["baseline", "canonical"])
def test_training_smoke_updates_model_and_records_validation_metrics(model_kind):
    tokenizer = local_fast_tokenizer()
    model = make_model(model_kind, tokenizer)
    before = model.classifier.weight.detach().clone()
    frame = example_frame()
    run_config = config(validation_interval_steps=1)

    result = fit_model(model, tokenizer, frame, validation_frame(), run_config, model_kind=model_kind)

    assert result.device == "cpu"
    assert result.effective_mixed_precision == "none"
    assert result.best_checkpoint_dir is None
    assert not torch.equal(before, model.classifier.weight.detach())
    assert result.best_validation_metrics["count"] == 4
    assert set(("loss", "accuracy", "precision", "recall", "f1", "confusion_matrix", "per_class_support")) <= set(result.best_validation_metrics)
    assert result.best_validation_metrics["confusion_matrix_labels"] == [0, 1]
    assert len([item for item in result.history if item["event"] == "validation"]) >= 1
    assert result.model.training


def test_gradient_accumulation_handles_partial_final_group_and_counts_optimizer_steps():
    tokenizer = local_fast_tokenizer()
    model = make_model("baseline", tokenizer)
    frame = example_frame().copy()
    frame.loc[len(frame)] = {
        "sentence": "one final sample",
        "label": 0,
        "idx": 20,
        "label_name": "Negative",
    }
    run_config = config(batch_size=2, gradient_accumulation_steps=2, epochs=1)

    result = fit_model(model, tokenizer, frame, validation_frame(), run_config, model_kind="baseline")

    epoch_record = next(item for item in result.history if item["event"] == "epoch")
    assert epoch_record["step"] == 2  # two optimizer updates for three microbatches
    assert torch.isfinite(torch.tensor(epoch_record["train_loss"]))


def test_epoch_boundary_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    import src.checkpointing as checkpointing_module

    tokenizer = local_fast_tokenizer()
    frame = example_frame().copy()
    frame.loc[len(frame)] = {
        "sentence": "one final sample",
        "label": 0,
        "idx": 20,
        "label_name": "Negative",
    }
    run_config = config(
        batch_size=2,
        gradient_accumulation_steps=2,
        epochs=2,
        validation_interval_steps=0,
    )
    provenance = {"source": {"train_sha256": "synthetic-source"}}

    torch.manual_seed(202)
    uninterrupted = make_model("canonical", tokenizer)
    initial_state = {name: value.detach().clone() for name, value in uninterrupted.state_dict().items()}
    full_result = fit_model(
        uninterrupted,
        tokenizer,
        frame,
        validation_frame(),
        run_config,
        model_kind="canonical",
        checkpoint_root=tmp_path / "continuous" / "checkpoints",
        dataset_provenance=provenance,
    )

    resumed_model = make_model("canonical", tokenizer)
    resumed_model.load_state_dict(initial_state)
    resume_root = tmp_path / "resumed" / "checkpoints"
    history_path = tmp_path / "resumed" / "training_history.json"
    original_publish = checkpointing_module.publish_checkpoint_bundle
    calls = 0

    def publish_then_interrupt(*args, **kwargs):
        nonlocal calls
        if kwargs.get("require_training_state"):
            calls += 1
            if calls == 1:
                raise RuntimeError("simulated interruption before checkpoint publication")
        return original_publish(*args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(checkpointing_module, "publish_checkpoint_bundle", publish_then_interrupt)
        with pytest.raises(RuntimeError, match="before checkpoint publication"):
            fit_model(
                resumed_model,
                tokenizer,
                frame,
                validation_frame(),
                run_config,
                model_kind="canonical",
                checkpoint_root=resume_root,
                dataset_provenance=provenance,
                history_path=history_path,
            )

    from src.checkpointing import resolve_resume_checkpoint

    interrupted_checkpoint = resolve_resume_checkpoint(resume_root)
    assert (interrupted_checkpoint / "training_state.pt").is_file()
    assert (resume_root / "latest_checkpoint.json").is_file()
    portable_root = tmp_path / "portable-copy" / "checkpoints"
    portable_root.mkdir(parents=True)
    portable_checkpoint = portable_root / interrupted_checkpoint.name
    shutil.copytree(interrupted_checkpoint, portable_checkpoint)
    history_path.write_text("not valid json", encoding="utf-8")
    relocated_config = replace(run_config, output_dir=str(portable_root.parent))
    resumed_result = fit_model(
        resumed_model,
        tokenizer,
        frame,
        validation_frame(),
        relocated_config,
        model_kind="canonical",
        checkpoint_root=portable_root,
        dataset_provenance=provenance,
        resume_checkpoint_dir=portable_checkpoint,
        history_path=history_path,
    )

    for name, expected in full_result.model.state_dict().items():
        torch.testing.assert_close(resumed_result.model.state_dict()[name], expected, rtol=0, atol=0)
    epoch_steps = [item["step"] for item in resumed_result.history if item["event"] == "epoch"]
    assert epoch_steps == [2, 4]
    assert resumed_result.best_checkpoint_dir is not None
    selected_best = Path(resumed_result.best_checkpoint_dir).resolve()
    assert selected_best.is_relative_to(portable_root.resolve())
    assert (selected_best / "metadata.json").is_file()
    assert (portable_checkpoint / "best_checkpoint" / "metadata.json").is_file()
    assert history_path.read_text(encoding="utf-8").startswith("[")


def test_cpu_resume_precision_metadata_mismatch_is_rejected():
    from src.training import execution_metadata

    current = execution_metadata(torch.device("cpu"), None)
    incompatible = {**current, "effective_precision": "float16"}
    with pytest.raises(ValueError, match="precision or device capabilities differ"):
        validate_resume_execution(incompatible, current)


def test_published_resume_snapshot_is_recovered_if_pointer_update_interrupted(tmp_path, monkeypatch):
    import src.checkpointing as checkpointing_module

    tokenizer = local_fast_tokenizer()
    run_config = config(epochs=1)
    root = tmp_path / "run" / "checkpoints"
    original_pointer = checkpointing_module.write_latest_resume_pointer

    def interrupt_pointer(*args, **kwargs):
        raise OSError("simulated interruption after checkpoint publication")

    monkeypatch.setattr(checkpointing_module, "write_latest_resume_pointer", interrupt_pointer)
    with pytest.raises(OSError, match="after checkpoint publication"):
        fit_model(
            make_model("baseline", tokenizer), tokenizer, example_frame(), validation_frame(), run_config,
            model_kind="baseline", checkpoint_root=root,
        )

    published = list(root.glob("resume-epoch-*"))
    assert len(published) == 1
    assert not (root / "latest_checkpoint.json").exists()
    (root / "latest_checkpoint.json").write_text("stale pointer", encoding="utf-8")
    monkeypatch.setattr(checkpointing_module, "write_latest_resume_pointer", original_pointer)
    recovered = checkpointing_module.recover_latest_resume_checkpoint(root)
    assert recovered == published[0]
    assert (root / "latest_checkpoint.json").is_file()
    corrupted_root = tmp_path / "corrupted-copy" / "checkpoints"
    shutil.copytree(root, corrupted_root)
    corrupted_snapshot = next(corrupted_root.glob("resume-epoch-*"))
    with (corrupted_snapshot / "training_state.pt").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.warns(RuntimeWarning, match="integrity hash"):
        assert checkpointing_module.recover_latest_resume_checkpoint(corrupted_root) is None
    tampered_best_root = tmp_path / "tampered-best-copy" / "checkpoints"
    shutil.copytree(root, tampered_best_root)
    tampered_snapshot = next(tampered_best_root.glob("resume-epoch-*"))
    with (tampered_snapshot / "best_checkpoint" / "model.safetensors").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.warns(RuntimeWarning, match="embedded best checkpoint fails its integrity hash"):
        assert checkpointing_module.recover_latest_resume_checkpoint(tampered_best_root) is None


def test_gradient_accumulation_matches_independent_full_batch_update(monkeypatch):
    """Three examples (2+1) must match a full-batch update within 1e-5."""
    tokenizer = local_fast_tokenizer()
    frame = example_frame().iloc[:3].copy()
    run_config = config(batch_size=2, gradient_accumulation_steps=2, epochs=1, learning_rate=3e-4)
    torch.manual_seed(912)
    accumulated_model = make_model("baseline", tokenizer)
    initial = {name: tensor.detach().clone() for name, tensor in accumulated_model.state_dict().items()}

    optimizer_steps = 0
    scheduler_steps = 0
    original_adamw = torch.optim.AdamW
    original_lambda_lr = torch.optim.lr_scheduler.LambdaLR

    class CountingAdamW(original_adamw):
        def step(self, *args, **kwargs):
            nonlocal optimizer_steps
            optimizer_steps += 1
            return super().step(*args, **kwargs)

    class CountingLambdaLR(original_lambda_lr):
        def __init__(self, *args, **kwargs):
            self._count_updates = False
            super().__init__(*args, **kwargs)
            self._count_updates = True

        def step(self, *args, **kwargs):
            nonlocal scheduler_steps
            if self._count_updates:
                scheduler_steps += 1
            return super().step(*args, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", CountingAdamW)
    monkeypatch.setattr(torch.optim.lr_scheduler, "LambdaLR", CountingLambdaLR)
    result = fit_model(
        accumulated_model, tokenizer, frame, validation_frame(), run_config, model_kind="baseline"
    )
    assert optimizer_steps == 1
    assert scheduler_steps == 1
    assert next(item for item in result.history if item["event"] == "epoch")["step"] == 1

    reference_model = make_model("baseline", tokenizer)
    reference_model.load_state_dict(initial)
    reference_model.train()
    batch = ClassificationCollator(tokenizer, run_config, "baseline")(
        [(row.sentence, int(row.label), int(row.idx)) for row in frame.itertuples()]
    )
    reference_optimizer = original_adamw(
        reference_model.parameters(), lr=run_config.learning_rate, weight_decay=run_config.weight_decay
    )
    reference_optimizer.zero_grad(set_to_none=True)
    logits = reference_model(**batch["inputs"]).logits
    torch.nn.functional.cross_entropy(logits, batch["labels"]).backward()
    torch.nn.utils.clip_grad_norm_(reference_model.parameters(), run_config.gradient_clip_norm)
    reference_optimizer.step()
    for name, expected in reference_model.state_dict().items():
        torch.testing.assert_close(
            accumulated_model.state_dict()[name], expected, rtol=1e-5, atol=1e-6
        )


def test_collator_uses_original_text_alignment_for_both_model_paths():
    tokenizer = local_fast_tokenizer()
    run_config = config()
    frame = example_frame().iloc[:2]
    for kind in ("canonical", "baseline"):
        loader = _build_loader(frame, tokenizer, run_config, kind, shuffle=False)
        batch = next(iter(loader))
        assert batch["labels"].tolist() == [1, 0]
        if kind == "canonical":
            aligned = batch["inputs"]
            assert aligned.source_texts == ("good movie", "bad film")
            assert aligned.unit_offsets[0][1:4] == ((0, 4), (4, 5), (5, 10))
        else:
            inputs = batch["inputs"]
            assert inputs["input_ids"].shape[0] == 2
            assert inputs["attention_mask"].sum(dim=1).tolist() == [4, 4]


def test_canonical_batch_provenance_survives_worker_style_serialization():
    tokenizer = local_fast_tokenizer()
    batch = collate_alignment_results([align_text("good movie", tokenizer)])
    transferred = pickle.loads(pickle.dumps(batch))
    model = make_model("canonical", tokenizer).eval()

    output = model(transferred)

    assert output.logits.shape == (1, 2)
    assert torch.isfinite(output.logits).all()


def test_malformed_serialized_batch_is_rejected_during_revalidation():
    batch = collate_alignment_results([align_text("good movie", local_fast_tokenizer())])
    offsets = batch.unit_offsets[0]
    malformed = replace(batch, unit_offsets=(offsets[:3] + ((4, 10),) + offsets[4:],))

    with pytest.raises(ValueError, match="do not form an ordered exact partition"):
        pickle.loads(pickle.dumps(malformed))


def test_validation_metrics_match_hand_computed_confusion_and_positive_scores():
    class FixedLogitModel(nn.Module):
        def forward(self, input_ids, attention_mask):
            predicted = input_ids[:, 0]
            logits = torch.full((len(predicted), 2), -5.0)
            logits.scatter_(1, predicted.unsqueeze(1), 5.0)
            return type("Out", (), {"logits": logits})()

    loader = [
        {
            "inputs": {
                "input_ids": torch.tensor([[0], [1], [0], [0]]),
                "attention_mask": torch.ones((4, 1), dtype=torch.long),
            },
            "labels": torch.tensor([0, 1, 1, 0]),
        }
    ]
    metrics = evaluate_loader(FixedLogitModel(), loader, device=torch.device("cpu"))

    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["per_class_support"] == {"0": 2, "1": 2}


def test_frame_validation_rejects_duplicate_text_and_bad_labels():
    frame = example_frame()
    with pytest.raises(ValueError, match="duplicate text"):
        validate_labeled_frame(frame.iloc[[0, 0]], "train")
    bad = example_frame()
    bad.loc[0, "label"] = 7
    with pytest.raises(ValueError, match="labels must"):
        validate_labeled_frame(bad, "train")


def test_requested_mixed_precision_falls_back_safely_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(RuntimeWarning, match="without CUDA"):
        assert resolve_mixed_precision("fp16", torch.device("cpu")) is None


def test_revisions_must_be_immutable_commit_shas():
    with pytest.raises(ValueError, match="exact 40-character commit SHA"):
        TrainingConfig(model_revision="main")


def test_asset_loader_uses_the_exact_local_only_model_and_tokenizer_revisions(monkeypatch):
    tokenizer = local_fast_tokenizer()
    backbone = tiny_backbone(len(tokenizer))
    calls = {}

    def tokenizer_loader(identifier, **kwargs):
        calls["tokenizer"] = (identifier, kwargs)
        return tokenizer

    def model_loader(cls, identifier, **kwargs):
        calls["model"] = (identifier, kwargs)
        return backbone

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(DistilBertModel, "from_pretrained", classmethod(model_loader))
    run_config = config()

    model, loaded_tokenizer = load_pretrained_components(
        run_config,
        model_kind="canonical",
        local_files_only=True,
    )

    assert loaded_tokenizer is tokenizer
    assert isinstance(model, CanonicalByteAwareDistilBert)
    assert calls["tokenizer"][0] == run_config.tokenizer_id
    assert calls["tokenizer"][1]["revision"] == run_config.tokenizer_revision
    assert calls["tokenizer"][1]["use_fast"] is True
    assert calls["tokenizer"][1]["local_files_only"] is True
    assert calls["model"][0] == run_config.model_id
    assert calls["model"][1] == {
        "revision": run_config.model_revision,
        "local_files_only": True,
    }
