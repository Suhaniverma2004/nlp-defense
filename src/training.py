"""Reproducible SST-2 training and clean evaluation for Milestone 2.3."""

from __future__ import annotations

import math
import json
import os
import random
import shutil
import uuid
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DistilBertConfig, DistilBertModel

from src.alignment import align_text
from src.canonical_model import AlignedModelBatch, CanonicalByteAwareDistilBert, collate_alignment_results
from src.sst2_data import LABEL_MAPPING
from src.unicode_policy import validate_text

PINNED_REPRODUCIBILITY_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
ModelKind = Literal["baseline", "canonical"]


@dataclass(frozen=True)
class TrainingConfig:
    """All run-defining settings, including immutable model/tokenizer revisions."""

    model_id: str = "distilbert-base-uncased"
    model_revision: str = PINNED_REPRODUCIBILITY_REVISION
    tokenizer_id: str = "distilbert-base-uncased"
    tokenizer_revision: str = PINNED_REPRODUCIBILITY_REVISION
    max_expanded_length: int = 128
    max_byte_span_length: int = 512
    batch_size: int = 16
    seed: int = 42
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    mixed_precision: Literal["none", "fp16", "bf16"] = "none"
    validation_interval_steps: int = 0
    gradient_accumulation_steps: int = 1
    num_workers: int = 0
    device: str = "auto"
    output_dir: str = "outputs/milestone2.3"
    byte_embedding_dim: int = 32
    byte_num_filters: int = 48
    byte_kernel_sizes: tuple[int, ...] = (2, 3, 4)
    byte_output_dim: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in ("model_id", "tokenizer_id", "output_dir"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string.")
        for name in ("model_revision", "tokenizer_revision"):
            revision = getattr(self, name)
            if not isinstance(revision, str) or len(revision) != 40 or any(
                character not in "0123456789abcdefABCDEF" for character in revision
            ):
                raise ValueError(f"{name} must be an exact 40-character commit SHA; mutable refs are not allowed.")
        for name in ("max_expanded_length", "max_byte_span_length", "batch_size", "epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if (
            isinstance(self.validation_interval_steps, bool)
            or not isinstance(self.validation_interval_steps, int)
            or isinstance(self.gradient_accumulation_steps, bool)
            or not isinstance(self.gradient_accumulation_steps, int)
            or self.gradient_accumulation_steps <= 0
            or isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or self.validation_interval_steps < 0
            or self.num_workers < 0
        ):
            raise ValueError("gradient_accumulation_steps must be positive; validation interval and workers must be non-negative.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite.")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative and finite.")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1).")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive and finite.")
        if self.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be none, fp16, or bf16.")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)) or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if not self.byte_kernel_sizes or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in self.byte_kernel_sizes
        ):
            raise ValueError("byte_kernel_sizes must contain positive integers.")
        for name in ("byte_embedding_dim", "byte_num_filters", "byte_output_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")


@dataclass(frozen=True)
class ClassifierOutput:
    logits: Tensor


class DistilBertBaselineClassifier(nn.Module):
    """DistilBERT-only SST-2 baseline with a standard two-layer task head."""

    def __init__(self, backbone: DistilBertModel, *, dropout: float = 0.1) -> None:
        super().__init__()
        if not isinstance(backbone, DistilBertModel):
            raise TypeError("backbone must be an explicitly constructed DistilBertModel.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.backbone = backbone
        self.pre_classifier = nn.Linear(backbone.config.dim, backbone.config.dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(backbone.config.dim, 2)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> ClassifierOutput:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have matching [batch, sequence] shapes.")
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = torch.relu(self.pre_classifier(output.last_hidden_state[:, 0]))
        return ClassifierOutput(self.classifier(self.dropout(pooled)))


class _FrameDataset(Dataset[tuple[str, int, int]]):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.texts = frame["sentence"].tolist()
        self.labels = frame["label"].astype("int64").tolist()
        if "source_row_index" in frame.columns:
            self.ids = frame["source_row_index"].tolist()
        else:
            self.ids = frame["idx"].tolist() if "idx" in frame.columns else list(range(len(frame)))

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, int, int]:
        return validate_text(self.texts[index], sample_id=self.ids[index]), self.labels[index], int(index)


def validate_labeled_frame(frame: pd.DataFrame, name: str) -> None:
    required = {"sentence", "label"}
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
        raise ValueError(f"{name} must be a DataFrame with sentence and label columns.")
    if frame.empty:
        raise ValueError(f"{name} must contain at least one labeled example.")
    if frame["sentence"].isna().any() or frame["label"].isna().any():
        raise ValueError(f"{name} contains missing sentence or label values.")
    if not pd.api.types.is_integer_dtype(frame["label"].dtype):
        raise ValueError(f"{name} labels must use an integer dtype with values 0 and 1.")
    if not set(frame["label"].tolist()).issubset(LABEL_MAPPING):
        raise ValueError(f"{name} labels must be integer values 0 and 1.")
    for i, text in enumerate(frame["sentence"].tolist()):
        validate_text(text, sample_id=f"{name}:{i}")
    if "label_name" in frame.columns:
        if not (frame["label_name"].reset_index(drop=True) == frame["label"].map(LABEL_MAPPING).reset_index(drop=True)).all():
            raise ValueError(f"{name} label_name values disagree with labels.")
    if frame["sentence"].duplicated().any():
        raise ValueError(f"{name} contains duplicate text; deduplicate before model selection.")


def seed_everything(seed: int) -> None:
    """Seed supported RNGs; deterministic kernels remain hardware-dependent."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu or auto.")
    return device


def resolve_mixed_precision(requested: str, device: torch.device) -> torch.dtype | None:
    """Return a CUDA autocast dtype, disabling unsupported requests explicitly."""
    if requested == "none":
        return None
    if device.type != "cuda" or not torch.cuda.is_available():
        warnings.warn("mixed precision requested without CUDA; disabling it for this run.", RuntimeWarning)
        return None
    if requested == "fp16":
        return torch.float16
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    with torch.cuda.device(device_index):
        bf16_supported = torch.cuda.is_bf16_supported()
    if not bf16_supported:
        warnings.warn("CUDA device does not support bfloat16; disabling mixed precision.", RuntimeWarning)
        return None
    return torch.bfloat16


def execution_metadata(device: torch.device, autocast_dtype: torch.dtype | None) -> dict[str, Any]:
    """Record effective precision and hardware capabilities for resume checks."""
    metadata: dict[str, Any] = {
        "resolved_device": str(device),
        "effective_precision": "none" if autocast_dtype is None else str(autocast_dtype).replace("torch.", ""),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": None,
        "cuda_capability": None,
        "cuda_bf16_supported": False,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        metadata["cuda_device_name"] = torch.cuda.get_device_name(index)
        metadata["cuda_capability"] = list(torch.cuda.get_device_capability(index))
        with torch.cuda.device(index):
            metadata["cuda_bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    return metadata


def validate_resume_execution(saved: Any, current: dict[str, Any]) -> None:
    """Reject resume when effective precision or device capability changed."""
    if not isinstance(saved, dict) or saved != current:
        raise ValueError(
            "resume checkpoint effective precision or device capabilities differ from this run; "
            "resume with a compatible device and precision configuration."
        )


def load_pretrained_components(
    config: TrainingConfig,
    *,
    model_kind: ModelKind,
    local_files_only: bool = False,
) -> tuple[nn.Module, Any]:
    """Load only the explicitly revision-pinned tokenizer and DistilBERT base."""
    if model_kind not in {"baseline", "canonical"}:
        raise ValueError("model_kind must be baseline or canonical.")
    # Seed before either classifier's randomly initialized task-specific layers
    # are constructed. Base-model tensors are loaded from the exact revision.
    seed_everything(config.seed)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_id,
            revision=config.tokenizer_revision,
            use_fast=True,
            local_files_only=local_files_only,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Could not load tokenizer {config.tokenizer_id}@{config.tokenizer_revision}; "
            "verify that exact revision is available locally or explicitly enable asset retrieval."
        ) from exc
    if getattr(tokenizer, "is_fast", False) is not True:
        raise RuntimeError("the configured tokenizer is not fast and cannot provide source offsets.")
    try:
        backbone = DistilBertModel.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            local_files_only=local_files_only,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Could not load DistilBERT {config.model_id}@{config.model_revision}; "
            "verify that exact revision is available locally or explicitly enable asset retrieval."
        ) from exc
    if config.max_expanded_length > backbone.config.max_position_embeddings:
        raise ValueError(
            f"max_expanded_length={config.max_expanded_length} exceeds the selected backbone's "
            f"position limit {backbone.config.max_position_embeddings}."
        )
    if backbone.config.vocab_size != len(tokenizer):
        raise ValueError(
            f"model vocabulary size {backbone.config.vocab_size} does not match tokenizer size {len(tokenizer)}."
        )
    if model_kind == "baseline":
        model: nn.Module = DistilBertBaselineClassifier(backbone, dropout=config.dropout)
    else:
        model = CanonicalByteAwareDistilBert(
            backbone,
            byte_embedding_dim=config.byte_embedding_dim,
            byte_num_filters=config.byte_num_filters,
            byte_kernel_sizes=config.byte_kernel_sizes,
            byte_output_dim=config.byte_output_dim,
            max_span_bytes=config.max_byte_span_length,
            dropout=config.dropout,
            pad_token_id=tokenizer.pad_token_id,
        )
    return model, tokenizer


class ClassificationCollator:
    """Use shared Milestone 2.1 alignments for canonical and baseline batches."""

    def __init__(self, tokenizer: Any, config: TrainingConfig, model_kind: ModelKind) -> None:
        if getattr(tokenizer, "is_fast", False) is not True:
            raise ValueError("a fast tokenizer with offset mappings is required.")
        if model_kind not in {"baseline", "canonical"}:
            raise ValueError("model_kind must be baseline or canonical.")
        self.tokenizer, self.config, self.model_kind = tokenizer, config, model_kind

    def __call__(self, rows: list[tuple[str, int, int]]) -> dict[str, Any]:
        if not rows:
            raise ValueError("cannot collate an empty batch.")
        alignments = [
            align_text(
                text,
                self.tokenizer,
                sample_id=sample_id,
                max_expanded_length=self.config.max_expanded_length,
                max_bytes_per_span=self.config.max_byte_span_length,
            )
            for text, _label, sample_id in rows
        ]
        labels = torch.tensor([label for _text, label, _sample_id in rows], dtype=torch.long)
        if self.model_kind == "canonical":
            return {"inputs": collate_alignment_results(alignments, pad_token_id=self.tokenizer.pad_token_id), "labels": labels}

        sequences = [torch.tensor(alignment.input_ids, dtype=torch.long) for alignment in alignments]
        width = max(sequence.numel() for sequence in sequences)
        input_ids = torch.full((len(sequences), width), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for index, sequence in enumerate(sequences):
            input_ids[index, : sequence.numel()] = sequence
            attention_mask[index, : sequence.numel()] = 1
        return {"inputs": {"input_ids": input_ids, "attention_mask": attention_mask}, "labels": labels}


def _build_loader(
    frame: pd.DataFrame,
    tokenizer: Any,
    config: TrainingConfig,
    model_kind: ModelKind,
    *,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed) if shuffle else None
    return DataLoader(
        _FrameDataset(frame),
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=config.num_workers,
        collate_fn=ClassificationCollator(tokenizer, config, model_kind),
        worker_init_fn=_seed_worker if config.num_workers else None,
    )


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _forward(model: nn.Module, inputs: Any) -> Tensor:
    if isinstance(inputs, AlignedModelBatch):
        return model(inputs).logits
    return model(inputs["input_ids"], inputs["attention_mask"]).logits


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Calculate clean SST-2 loss and binary classification metrics."""
    model.eval()
    losses = 0.0
    count = 0
    labels_all: list[int] = []
    predictions_all: list[int] = []
    for batch in loader:
        labels = batch["labels"].to(device)
        inputs = batch["inputs"]
        if isinstance(inputs, AlignedModelBatch):
            inputs = inputs.to(device)
        else:
            inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            logits = _forward(model, inputs)
            loss = nn.functional.cross_entropy(logits.float(), labels, reduction="sum")
        if not torch.isfinite(loss) or not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite logits or validation loss encountered.")
        losses += float(loss.item())
        count += int(labels.numel())
        labels_all.extend(labels.cpu().tolist())
        predictions_all.extend(logits.argmax(dim=-1).cpu().tolist())
    if count == 0:
        raise ValueError("evaluation loader produced no examples.")
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_all, predictions_all, labels=[1], average=None, zero_division=0
    )
    matrix = confusion_matrix(labels_all, predictions_all, labels=[0, 1])
    return {
        "count": count,
        "loss": losses / count,
        "accuracy": sum(a == b for a, b in zip(labels_all, predictions_all)) / count,
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": matrix.tolist(),
        "per_class_support": {str(label): int((np.asarray(labels_all) == label).sum()) for label in (0, 1)},
        "positive_class_support": int(support[0]),
    }


@dataclass
class TrainingResult:
    model: nn.Module
    history: list[dict[str, Any]]
    best_validation_metrics: dict[str, Any]
    best_checkpoint_dir: str | None
    device: str
    effective_mixed_precision: str


def fit_model(
    model: nn.Module,
    tokenizer: Any,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    config: TrainingConfig,
    *,
    model_kind: ModelKind,
    checkpoint_root: str | Path | None = None,
    dataset_provenance: dict[str, Any] | None = None,
    resume_checkpoint_dir: str | Path | None = None,
    history_path: str | Path | None = None,
) -> TrainingResult:
    """Train on the training frame; validation alone drives checkpoint selection.

    No test-set argument exists by design. Exact source text flows through
    ``align_text`` and the canonical collator; all labels/losses are external.
    """
    validate_labeled_frame(train_frame, "train")
    validate_labeled_frame(validation_frame, "validation")
    overlap = set(train_frame["sentence"]) & set(validation_frame["sentence"])
    if overlap:
        raise ValueError(f"train and validation contain {len(overlap)} exact-text overlaps.")
    if model_kind not in {"baseline", "canonical"}:
        raise ValueError("model_kind must be baseline or canonical.")

    if resume_checkpoint_dir is not None and checkpoint_root is None:
        raise ValueError("checkpoint_root is required when resuming a training run.")

    seed_everything(config.seed)
    device = resolve_device(config.device)
    autocast_dtype = resolve_mixed_precision(config.mixed_precision, device)
    effective_precision = "none" if autocast_dtype is None else str(autocast_dtype).replace("torch.", "")
    model.to(device)
    train_loader = _build_loader(train_frame, tokenizer, config, model_kind, shuffle=True)
    validation_loader = _build_loader(validation_frame, tokenizer, config, model_kind, shuffle=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config.weight_decay,
        amsgrad=False,
    )
    microbatches_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = math.ceil(microbatches_per_epoch / config.gradient_accumulation_steps)
    total_steps = max(1, config.epochs * optimizer_steps_per_epoch)
    warmup_steps = int(total_steps * config.warmup_ratio)

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        decay_steps = max(1, total_steps - warmup_steps)
        return max(0.0, float(total_steps - step) / decay_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=autocast_dtype == torch.float16)
    history: list[dict[str, Any]] = []
    best_metrics: dict[str, Any] | None = None
    best_loss = float("inf")
    best_path: str | None = None
    global_step = 0
    last_validation_step = -1
    start_epoch = 0

    if resume_checkpoint_dir is not None:
        from src.checkpointing import (
            load_checkpoint,
            load_training_state,
            resolve_resume_checkpoint,
            restore_rng_state,
            runtime_software_versions,
            validate_checkpoint_bundle,
        )

        resume_checkpoint_dir = resolve_resume_checkpoint(resume_checkpoint_dir)

        resume_model, resume_tokenizer, resume_metadata = load_checkpoint(
            resume_checkpoint_dir,
            expected_model_kind=model_kind,
            expected_model_id=config.model_id,
            expected_model_revision=config.model_revision,
            expected_tokenizer_id=config.tokenizer_id,
            expected_tokenizer_revision=config.tokenizer_revision,
            device=device,
        )
        if resume_metadata.get("dataset_provenance") != (dataset_provenance or {}):
            raise ValueError("resume checkpoint dataset provenance does not match the current split manifest.")
        if resume_metadata.get("software_versions") != runtime_software_versions():
            raise ValueError("resume checkpoint software versions differ from the current runtime.")
        current_execution = execution_metadata(device, autocast_dtype)
        validate_resume_execution(resume_metadata.get("execution"), current_execution)
        saved_config = dict(resume_metadata["training_config"])
        current_config = json.loads(json.dumps(asdict(config)))
        saved_config.pop("output_dir", None)
        current_config.pop("output_dir", None)
        if saved_config != current_config:
            raise ValueError("resume checkpoint training configuration differs from the requested run.")
        if not getattr(resume_tokenizer, "is_fast", False) or resume_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise ValueError("resume checkpoint tokenizer does not match the supplied training tokenizer.")
        model.load_state_dict(resume_model.state_dict(), strict=True)
        state = load_training_state(Path(resume_checkpoint_dir) / "training_state.pt", trusted=True)
        resume_extra = state.get("extra_state", {})
        validate_resume_execution(resume_extra.get("execution"), current_execution)
        if int(state.get("epoch", 0)) < 1 or int(state["epoch"]) > config.epochs:
            raise ValueError("resume checkpoint epoch is outside the configured training run.")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        saved_scaler_state = state.get("scaler")
        if scaler.is_enabled() and saved_scaler_state:
            scaler.load_state_dict(saved_scaler_state)
        elif scaler.is_enabled() or saved_scaler_state not in (None, {}):
            raise ValueError("resume checkpoint mixed-precision scaler does not match the current run.")
        start_epoch = int(state["epoch"])
        global_step = int(state["global_step"])
        last_validation_step = global_step
        best_metrics = resume_extra.get("best_validation_metrics")
        best_loss = float(resume_extra.get("best_validation_loss", float("inf")))
        best_relative = resume_extra.get("best_checkpoint_relative")
        if best_relative != "best_checkpoint":
            raise ValueError("resume checkpoint has no portable embedded best-checkpoint bundle.")
        embedded_best = Path(resume_checkpoint_dir) / best_relative
        validate_checkpoint_bundle(embedded_best)
        best_path = str(embedded_best)
        restore_rng_state(state, data_loader_generator=train_loader.generator)
        if isinstance(resume_extra.get("history"), list):
            history = resume_extra["history"]
        else:
            raise ValueError("resume snapshot has no recoverable epoch-boundary history.")
        if history_path is not None:
            # Snapshot history is authoritative at an epoch boundary; this
            # discards records from any later, incomplete epoch and repairs a
            # torn or corrupted history file.
            save_training_history(history, history_path)

    def persist_history() -> None:
        if history_path is not None:
            save_training_history(history, history_path)

    def validate_and_maybe_save(epoch_number: int) -> dict[str, Any]:
        nonlocal best_metrics, best_loss, best_path, last_validation_step
        was_training = model.training
        try:
            metrics = evaluate_loader(model, validation_loader, device=device, autocast_dtype=autocast_dtype)
        finally:
            if was_training:
                model.train()
        record = {"event": "validation", "epoch": epoch_number, "step": global_step, **metrics}
        history.append(record)
        persist_history()
        last_validation_step = global_step
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_metrics = metrics
            if checkpoint_root is not None:
                from src.checkpointing import save_checkpoint

                checkpoint_path = Path(checkpoint_root) / (
                    f"checkpoint-step-{global_step:08d}-{uuid.uuid4().hex[:8]}"
                )
                save_checkpoint(
                    model,
                    tokenizer,
                    checkpoint_path,
                    model_kind=model_kind,
                    training_config=asdict(config),
                    dataset_provenance=dataset_provenance or {},
                    best_validation_metrics=metrics,
                    execution_metadata=execution_metadata(device, autocast_dtype),
                )
                best_path = str(checkpoint_path)
        return metrics

    for epoch_index in range(start_epoch, config.epochs):
        model.train()
        running_loss = 0.0
        observed = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader):
            labels = batch["labels"].to(device)
            inputs = batch["inputs"]
            if isinstance(inputs, AlignedModelBatch):
                inputs = inputs.to(device)
            else:
                inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                logits = _forward(model, inputs)
                loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise FloatingPointError("non-finite logits or training loss encountered.")
            group_start = (batch_index // config.gradient_accumulation_steps) * config.gradient_accumulation_steps
            group_end = min(group_start + config.gradient_accumulation_steps, microbatches_per_epoch)
            group_examples = sum(
                min(config.batch_size, len(train_frame) - micro_index * config.batch_size)
                for micro_index in range(group_start, group_end)
            )
            scaled_loss = loss * (int(labels.numel()) / group_examples)
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            is_update_boundary = (
                (batch_index + 1) % config.gradient_accumulation_steps == 0
                or batch_index + 1 == microbatches_per_epoch
            )
            if is_update_boundary:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_updated = scaler.get_scale() >= scale_before
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                    optimizer.step()
                    optimizer_updated = True
                optimizer.zero_grad(set_to_none=True)
                if optimizer_updated:
                    scheduler.step()
                    global_step += 1
                    if config.validation_interval_steps and global_step % config.validation_interval_steps == 0:
                        validate_and_maybe_save(epoch_index + 1)
            running_loss += float(loss.item()) * int(labels.numel())
            observed += int(labels.numel())
        if last_validation_step != global_step:
            validate_and_maybe_save(epoch_index + 1)
        history.append(
            {
                "event": "epoch",
                "epoch": epoch_index + 1,
                "step": global_step,
                "train_loss": running_loss / observed,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        persist_history()
        if checkpoint_root is not None:
            from src.checkpointing import save_checkpoint, save_training_state

            checkpoint_root = Path(checkpoint_root)
            resume_path = checkpoint_root / (
                f"resume-epoch-{epoch_index + 1:04d}-step-{global_step:08d}-{uuid.uuid4().hex[:8]}"
            )
            staging_path = save_checkpoint(
                model,
                tokenizer,
                resume_path,
                model_kind=model_kind,
                training_config=asdict(config),
                dataset_provenance=dataset_provenance or {},
                best_validation_metrics=best_metrics or {},
                execution_metadata=execution_metadata(device, autocast_dtype),
                publish=False,
            )
            if best_path is None:
                raise RuntimeError("cannot publish resumable state without a selected best-validation checkpoint.")
            from src.checkpointing import (
                publish_checkpoint_bundle,
                seal_training_state_bundle,
                write_latest_resume_pointer,
            )

            shutil.copytree(best_path, staging_path / "best_checkpoint")
            save_training_state(
                staging_path / "training_state.pt",
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch_index + 1,
                global_step=global_step,
                scaler=scaler,
                data_loader_generator=train_loader.generator,
                extra_state={
                    "best_validation_metrics": best_metrics or {},
                    "best_validation_loss": best_loss,
                    "best_checkpoint_relative": "best_checkpoint",
                    "history": history,
                    "execution": execution_metadata(device, autocast_dtype),
                },
            )
            seal_training_state_bundle(staging_path)
            publish_checkpoint_bundle(staging_path, resume_path, require_training_state=True)
            write_latest_resume_pointer(checkpoint_root, resume_path, epoch_index + 1, global_step)
    if best_metrics is None:
        raise RuntimeError("training completed without a validation result.")
    return TrainingResult(model, history, best_metrics, best_path, str(device), effective_precision)


def save_training_history(history: list[dict[str, Any]], path: str | Path) -> None:
    """Atomically publish run history; JSON is the authoritative commit marker."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    json_stage = path.parent / f".{path.name}.tmp-{nonce}"
    csv_path = path.with_suffix(".csv")
    csv_stage = path.parent / f".{csv_path.name}.tmp-{nonce}"
    json_stage.write_text(json_dumps(history), encoding="utf-8")
    pd.json_normalize(history).to_csv(csv_stage, index=False)
    # Publish CSV first; JSON signals that both artifacts were prepared.
    os.replace(csv_stage, csv_path)
    os.replace(json_stage, path)


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
