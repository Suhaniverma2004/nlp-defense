"""Versioned, integrity-checked safetensors checkpoints for both SST-2 models."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import tempfile
import uuid
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import safetensors
import sklearn
import torch
from safetensors.torch import load_file, save_file
import tokenizers
import transformers
from transformers import AutoTokenizer, DistilBertConfig, DistilBertModel

from src.canonical_model import CanonicalByteAwareDistilBert
from src.sst2_data import LABEL_MAPPING
from src.training import DistilBertBaselineClassifier

CHECKPOINT_SCHEMA = "nlp-defense-sst2"
CHECKPOINT_VERSION = 1
ALIGNMENT_CONTRACT_VERSION = "milestone-2.1"
BYTE_ENCODING_CONTRACT = {"name": "utf8-byte-plus-one", "offset": 1, "pad_id": 0, "valid_ids": [1, 256]}
ModelKind = Literal["baseline", "canonical"]


class CheckpointError(ValueError):
    """Raised when checkpoint files or experiment metadata are incompatible."""


def runtime_software_versions() -> dict[str, str]:
    """Return versions relevant to reproducing model and training state behavior."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": __import__("pandas").__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "safetensors": safetensors.__version__,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokenizer_manifest(root: Path) -> dict[str, str]:
    return {
        item.relative_to(root).as_posix(): _sha256(item)
        for item in sorted(root.rglob("*"))
        if item.is_file()
    }


def _directory_sha256(root: Path) -> str:
    manifest = {
        item.relative_to(root).as_posix(): _sha256(item)
        for item in sorted(root.rglob("*"))
        if item.is_file()
    }
    serialized = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _architecture_config(model: Any, model_kind: ModelKind) -> dict[str, Any]:
    backbone = model.backbone
    if model_kind == "baseline" and isinstance(model, DistilBertBaselineClassifier):
        return {
            "backbone": backbone.config.to_dict(),
            "dropout": float(model.dropout.p),
            "num_labels": 2,
            "head": "distilbert-sequence-classification-v1",
        }
    if model_kind == "canonical" and isinstance(model, CanonicalByteAwareDistilBert):
        return {
            "backbone": backbone.config.to_dict(),
            "byte_embedding_dim": model.byte_encoder.embedding_dim,
            "byte_num_filters": model.byte_encoder.num_filters,
            "byte_kernel_sizes": list(model.byte_encoder.kernel_sizes),
            "byte_output_dim": model.byte_encoder.output_dim,
            "max_span_bytes": model.byte_encoder.max_span_bytes,
            "dropout": float(model.classifier_dropout.p),
            "pad_token_id": model.pad_token_id,
            "num_labels": 2,
            "fusion_gate_shape": list(model.fusion_gate.shape),
        }
    raise TypeError(f"model instance does not match model_kind={model_kind!r}.")


def save_checkpoint(
    model: Any,
    tokenizer: Any,
    output_dir: str | Path,
    *,
    model_kind: ModelKind,
    training_config: dict[str, Any],
    dataset_provenance: dict[str, Any],
    best_validation_metrics: dict[str, Any],
    execution_metadata: dict[str, Any] | None = None,
    publish: bool = True,
) -> Path:
    """Build a complete checkpoint in a unique sibling directory.

    With ``publish=False``, return the private staging directory so a caller can
    add resume state before publishing the whole bundle.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"immutable checkpoint destination already exists: {output_dir}")
    if getattr(tokenizer, "is_fast", False) is not True:
        raise CheckpointError("checkpoint tokenizer must be a fast tokenizer.")
    if model_kind not in {"baseline", "canonical"}:
        raise CheckpointError("model_kind must be baseline or canonical.")
    architecture = _architecture_config(model, model_kind)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.tmp-",
        dir=output_dir.parent,
    ))
    tokenizer_dir = staging_dir / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer.save_pretrained(tokenizer_dir)

    weights_path = staging_dir / "model.safetensors"
    weights = {
        name: tensor.detach().to(device="cpu").contiguous()
        for name, tensor in model.state_dict().items()
    }
    save_file(weights, str(weights_path))
    metadata = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_VERSION,
        "software_versions": runtime_software_versions(),
        "execution": execution_metadata,
        "model_kind": model_kind,
        "model_id": training_config["model_id"],
        "model_revision": training_config["model_revision"],
        "tokenizer_id": training_config["tokenizer_id"],
        "tokenizer_revision": training_config["tokenizer_revision"],
        "tokenizer_is_fast": True,
        "tokenizer_files_sha256": _tokenizer_manifest(tokenizer_dir),
        "weights_file": weights_path.name,
        "weights_sha256": _sha256(weights_path),
        "architecture": architecture,
        "label_mapping": {str(key): value for key, value in LABEL_MAPPING.items()},
        "byte_encoding": BYTE_ENCODING_CONTRACT,
        "alignment_contract_version": ALIGNMENT_CONTRACT_VERSION,
        "max_expanded_length": int(training_config["max_expanded_length"]),
        "max_byte_span_length": int(training_config["max_byte_span_length"]),
        "dataset_provenance": dataset_provenance,
        "training_config": training_config,
        "best_validation_metrics": best_validation_metrics,
    }
    metadata_path = staging_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    validate_checkpoint_bundle(staging_dir)
    if not publish:
        return staging_dir
    publish_checkpoint_bundle(staging_dir, output_dir)
    return output_dir


def validate_checkpoint_bundle(
    checkpoint_dir: str | Path,
    *,
    require_training_state: bool = False,
) -> dict[str, Any]:
    """Validate the complete metadata, tokenizer, weights, and optional resume state."""
    root = Path(checkpoint_dir)
    metadata = _read_metadata(root)
    tokenizer_dir = root / "tokenizer"
    if not tokenizer_dir.is_dir() or _tokenizer_manifest(tokenizer_dir) != metadata.get("tokenizer_files_sha256"):
        raise CheckpointError(f"checkpoint tokenizer files are incomplete or invalid in {root}.")
    weights_path = root / "model.safetensors"
    if metadata.get("weights_file") != "model.safetensors" or not weights_path.is_file():
        raise CheckpointError(f"checkpoint model weights are missing in {root}.")
    if _sha256(weights_path) != metadata.get("weights_sha256"):
        raise CheckpointError(f"checkpoint model weights fail their integrity hash in {root}.")
    if require_training_state:
        training_state_path = root / "training_state.pt"
        if not training_state_path.is_file() or _sha256(training_state_path) != metadata.get("training_state_sha256"):
            raise CheckpointError(f"checkpoint training state is missing or fails its integrity hash in {root}.")
        state = load_training_state(training_state_path, trusted=True)
        if (
            not isinstance(state.get("optimizer"), dict)
            or not {"state", "param_groups"}.issubset(state["optimizer"])
            or not isinstance(state.get("scheduler"), dict)
            or not isinstance(state.get("scaler"), (dict, type(None)))
            or not isinstance(state.get("epoch"), int)
            or not isinstance(state.get("global_step"), int)
        ):
            raise CheckpointError(f"checkpoint training state is incomplete in {root}.")
        required_rng = {"python", "numpy", "torch_cpu", "torch_cuda", "data_loader_generator"}
        if not required_rng.issubset(state["rng"]):
            raise CheckpointError(f"checkpoint training RNG state is incomplete in {root}.")
        rng = state["rng"]
        numpy_state = rng["numpy"]
        if (
            not isinstance(rng["python"], tuple)
            or not isinstance(numpy_state, dict)
            or not {"name", "keys", "position", "has_gauss", "cached_gaussian"}.issubset(numpy_state)
            or not isinstance(rng["torch_cpu"], torch.Tensor)
            or not isinstance(rng["torch_cuda"], list)
            or any(not isinstance(item, torch.Tensor) for item in rng["torch_cuda"])
            or (rng["data_loader_generator"] is not None and not isinstance(rng["data_loader_generator"], torch.Tensor))
            or int(state["epoch"]) <= 0
            or int(state["global_step"]) < 0
        ):
            raise CheckpointError(f"checkpoint training RNG or progress state is malformed in {root}.")
        best_dir = root / "best_checkpoint"
        if _directory_sha256(best_dir) != metadata.get("embedded_best_checkpoint_sha256"):
            raise CheckpointError(f"embedded best checkpoint fails its integrity hash in {root}.")
        best_metadata = validate_checkpoint_bundle(best_dir)
        for field in ("model_kind", "model_id", "model_revision", "tokenizer_id", "tokenizer_revision"):
            if best_metadata.get(field) != metadata.get(field):
                raise CheckpointError(f"embedded best checkpoint {field} does not match resume bundle {root}.")
        if best_metadata.get("dataset_provenance") != metadata.get("dataset_provenance"):
            raise CheckpointError(f"embedded best checkpoint dataset provenance differs in {root}.")
        best_config = dict(best_metadata.get("training_config") or {})
        resume_config = dict(metadata.get("training_config") or {})
        best_config.pop("output_dir", None)
        resume_config.pop("output_dir", None)
        if json.dumps(best_config, sort_keys=True) != json.dumps(resume_config, sort_keys=True):
            raise CheckpointError(f"embedded best checkpoint training configuration differs in {root}.")
        if best_metadata.get("execution") != metadata.get("execution"):
            raise CheckpointError(f"embedded best checkpoint execution metadata differs in {root}.")
        best_metrics = state["extra_state"].get("best_validation_metrics")
        state_best_loss = state["extra_state"].get("best_validation_loss")
        if json.dumps(best_metrics, sort_keys=True) != json.dumps(
            best_metadata.get("best_validation_metrics"), sort_keys=True
        ):
            raise CheckpointError(f"embedded best checkpoint metrics differ from resume state in {root}.")
        if not isinstance(best_metrics, dict) or best_metrics.get("loss") != state_best_loss:
            raise CheckpointError(f"embedded best checkpoint selection loss differs from resume state in {root}.")
    return metadata


def seal_training_state_bundle(checkpoint_dir: str | Path) -> None:
    """Hash a completed training-state file into staging metadata before publication."""
    root = Path(checkpoint_dir)
    metadata_path = root / "metadata.json"
    metadata = _read_metadata(root)
    training_state_path = root / "training_state.pt"
    if not training_state_path.is_file():
        raise CheckpointError("cannot seal a resume bundle without training_state.pt.")
    metadata["training_state_sha256"] = _sha256(training_state_path)
    best_dir = root / "best_checkpoint"
    if not best_dir.is_dir():
        raise CheckpointError("cannot seal a resume bundle without its embedded best checkpoint.")
    metadata["embedded_best_checkpoint_sha256"] = _directory_sha256(best_dir)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def publish_checkpoint_bundle(
    staging_dir: str | Path,
    output_dir: str | Path,
    *,
    require_training_state: bool = False,
) -> Path:
    """Publish a validated sibling staging directory without replacing artifacts."""
    staging_dir, output_dir = Path(staging_dir), Path(output_dir)
    if staging_dir.parent.resolve() != output_dir.parent.resolve():
        raise CheckpointError("staging and final checkpoint directories must share a parent filesystem.")
    if output_dir.exists():
        raise FileExistsError(f"immutable checkpoint destination already exists: {output_dir}")
    validate_checkpoint_bundle(staging_dir, require_training_state=require_training_state)
    try:
        os.replace(staging_dir, output_dir)
    except OSError as exc:
        raise CheckpointError(
            f"checkpoint could not be atomically published from {staging_dir} to {output_dir}; "
            "the staging directory has been preserved for recovery."
        ) from exc
    return output_dir


def _read_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    try:
        metadata = json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"checkpoint metadata is missing or invalid in {checkpoint_dir}.") from exc
    if metadata.get("schema") != CHECKPOINT_SCHEMA or metadata.get("schema_version") != CHECKPOINT_VERSION:
        raise CheckpointError("unsupported checkpoint schema or version.")
    training_config = metadata.get("training_config")
    if not isinstance(training_config, dict):
        raise CheckpointError("checkpoint training configuration is missing or invalid.")
    for field in ("model_id", "model_revision", "tokenizer_id", "tokenizer_revision"):
        if not metadata.get(field) or training_config.get(field) != metadata[field]:
            raise CheckpointError(f"checkpoint {field} metadata disagrees with its training configuration.")
    for field in ("model_revision", "tokenizer_revision"):
        revision = metadata[field]
        if not isinstance(revision, str) or len(revision) != 40 or any(
            character not in "0123456789abcdefABCDEF" for character in revision
        ):
            raise CheckpointError(f"checkpoint {field} is not an immutable commit SHA.")
    if metadata.get("model_kind") not in {"baseline", "canonical"}:
        raise CheckpointError("checkpoint model kind is unsupported.")
    for field in ("max_expanded_length", "max_byte_span_length"):
        if not isinstance(metadata.get(field), int) or metadata[field] <= 0:
            raise CheckpointError(f"checkpoint {field} is missing or invalid.")
        if training_config.get(field) != metadata[field]:
            raise CheckpointError(f"checkpoint {field} disagrees with its training configuration.")
    if metadata.get("label_mapping") != {str(key): value for key, value in LABEL_MAPPING.items()}:
        raise CheckpointError("checkpoint label mapping is incompatible with SST-2 labels.")
    if metadata.get("byte_encoding") != BYTE_ENCODING_CONTRACT:
        raise CheckpointError("checkpoint byte-ID mapping or padding contract is incompatible.")
    if metadata.get("alignment_contract_version") != ALIGNMENT_CONTRACT_VERSION:
        raise CheckpointError("checkpoint alignment contract version is incompatible.")
    return metadata


def load_checkpoint(
    checkpoint_dir: str | Path,
    *,
    expected_model_kind: ModelKind | None = None,
    expected_model_id: str | None = None,
    expected_model_revision: str | None = None,
    expected_tokenizer_id: str | None = None,
    expected_tokenizer_revision: str | None = None,
    device: str | torch.device = "cpu",
) -> tuple[Any, Any, dict[str, Any]]:
    """Load only packaged local tokenizer assets and strict safetensors weights."""
    root = Path(checkpoint_dir)
    metadata = _read_metadata(root)
    expected_values = {
        "model_kind": expected_model_kind,
        "model_id": expected_model_id,
        "model_revision": expected_model_revision,
        "tokenizer_id": expected_tokenizer_id,
        "tokenizer_revision": expected_tokenizer_revision,
    }
    for field, expected in expected_values.items():
        if expected is not None and metadata.get(field) != expected:
            raise CheckpointError(
                f"checkpoint {field} {metadata.get(field)!r} does not match expected {expected!r}."
            )

    tokenizer_dir = root / "tokenizer"
    actual_tokenizer_files = _tokenizer_manifest(tokenizer_dir) if tokenizer_dir.is_dir() else {}
    if actual_tokenizer_files != metadata.get("tokenizer_files_sha256"):
        raise CheckpointError("packaged tokenizer files are missing or fail their integrity hashes.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True, local_files_only=True)
    except (OSError, ValueError) as exc:
        raise CheckpointError("packaged tokenizer could not be loaded offline.") from exc
    if getattr(tokenizer, "is_fast", False) is not True:
        raise CheckpointError("packaged tokenizer is not fast.")

    if metadata.get("weights_file") != "model.safetensors":
        raise CheckpointError("checkpoint weights_file must be the packaged model.safetensors artifact.")
    weights_path = root / "model.safetensors"
    if not weights_path.is_file() or _sha256(weights_path) != metadata.get("weights_sha256"):
        raise CheckpointError("checkpoint model weights are missing or fail their integrity hash.")
    architecture = metadata.get("architecture", {})
    try:
        backbone_config = DistilBertConfig.from_dict(architecture["backbone"])
        if backbone_config.model_type != "distilbert":
            raise CheckpointError("checkpoint backbone config is not DistilBERT.")
        if backbone_config.vocab_size != len(tokenizer):
            raise CheckpointError(
                f"checkpoint model vocabulary size {backbone_config.vocab_size} does not match "
                f"packaged tokenizer size {len(tokenizer)}."
            )
        if metadata["max_expanded_length"] > backbone_config.max_position_embeddings:
            raise CheckpointError(
                "checkpoint expanded-sequence limit exceeds the saved backbone position limit."
            )
        backbone = DistilBertModel(backbone_config)
        model_kind = metadata["model_kind"]
        if model_kind == "baseline":
            model = DistilBertBaselineClassifier(backbone, dropout=float(architecture["dropout"]))
        elif model_kind == "canonical":
            if architecture.get("fusion_gate_shape") != []:
                raise CheckpointError("canonical checkpoint fusion gate must be scalar.")
            model = CanonicalByteAwareDistilBert(
                backbone,
                byte_embedding_dim=int(architecture["byte_embedding_dim"]),
                byte_num_filters=int(architecture["byte_num_filters"]),
                byte_kernel_sizes=tuple(architecture["byte_kernel_sizes"]),
                byte_output_dim=int(architecture["byte_output_dim"]),
                max_span_bytes=architecture["max_span_bytes"],
                dropout=float(architecture["dropout"]),
                pad_token_id=int(architecture["pad_token_id"]),
            )
        else:
            raise CheckpointError(f"unsupported checkpoint model kind: {model_kind!r}.")
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, CheckpointError):
            raise
        raise CheckpointError("checkpoint architecture metadata is invalid.") from exc

    try:
        state = load_file(str(weights_path), device="cpu")
        model.load_state_dict(state, strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise CheckpointError(f"checkpoint weights failed strict loading: {exc}") from exc
    model.to(device)
    model.eval()
    return model, tokenizer, metadata


def save_training_state(
    path: str | Path,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
    scaler: Any | None = None,
    data_loader_generator: torch.Generator | None = None,
    extra_state: dict[str, Any] | None = None,
) -> None:
    """Save a trusted epoch-boundary training snapshot and all RNG states."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy_name, numpy_keys, numpy_pos, numpy_has_gauss, numpy_cached = np.random.get_state()
    cuda_states = [state.cpu() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    state = {
        "training_state_version": 2,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": {
            "python": random.getstate(),
            "numpy": {
                "name": numpy_name,
                "keys": numpy_keys.astype(np.uint32).tolist(),
                "position": int(numpy_pos),
                "has_gauss": int(numpy_has_gauss),
                "cached_gaussian": float(numpy_cached),
            },
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": cuda_states,
            "data_loader_generator": (
                data_loader_generator.get_state() if data_loader_generator is not None else None
            ),
        },
        "extra_state": extra_state or {},
    }
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_training_state(path: str | Path, *, trusted: bool = False) -> dict[str, Any]:
    """Load a separate trusted training-state artifact (never a model checkpoint)."""
    if not trusted:
        raise CheckpointError("training-state files use PyTorch serialization; pass trusted=True only for trusted artifacts.")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise CheckpointError("trusted training-state artifact could not be loaded safely.") from exc
    if not isinstance(state, dict) or state.get("training_state_version") != 2:
        raise CheckpointError("unsupported training-state schema.")
    if not isinstance(state.get("rng"), dict) or not isinstance(state.get("extra_state"), dict):
        raise CheckpointError("training-state RNG or supplemental metadata is missing.")
    return state


def restore_rng_state(
    state: dict[str, Any],
    *,
    data_loader_generator: torch.Generator | None = None,
) -> None:
    """Restore Python, NumPy, Torch CPU/CUDA, and optional loader RNG state."""
    rng = state.get("rng")
    if not isinstance(rng, dict):
        raise CheckpointError("training-state RNG data is missing or invalid.")
    try:
        random.setstate(rng["python"])
        numpy_state = rng["numpy"]
        np.random.set_state((
            numpy_state["name"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        ))
        torch.set_rng_state(rng["torch_cpu"].cpu())
        cuda_states = rng["torch_cuda"]
        if cuda_states:
            if not torch.cuda.is_available():
                raise CheckpointError("checkpoint contains CUDA RNG state but CUDA is unavailable.")
            if len(cuda_states) != torch.cuda.device_count():
                raise CheckpointError("CUDA device count differs from the saved training RNG state.")
            torch.cuda.set_rng_state_all(cuda_states)
        loader_state = rng.get("data_loader_generator")
        if loader_state is not None:
            if data_loader_generator is None:
                raise CheckpointError("checkpoint has DataLoader RNG state but no loader generator was supplied.")
            data_loader_generator.set_state(loader_state.cpu())
    except CheckpointError:
        raise
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise CheckpointError(f"training-state RNG could not be restored: {exc}") from exc


_RESUME_NAME = re.compile(r"^resume-epoch-(\d+)-step-(\d+)-([0-9a-f]{8})$")


def _validate_resume_snapshot(path: Path, *, expected_name: str | None = None) -> tuple[int, int]:
    name = expected_name or path.name
    match = _RESUME_NAME.fullmatch(name)
    if match is None:
        raise CheckpointError(f"resume snapshot name is not versioned by epoch and step: {name!r}.")
    epoch_expected, step_expected = int(match.group(1)), int(match.group(2))
    validate_checkpoint_bundle(path, require_training_state=True)
    state = load_training_state(path / "training_state.pt", trusted=True)
    epoch, step = int(state["epoch"]), int(state["global_step"])
    if (epoch, step) != (epoch_expected, step_expected):
        raise CheckpointError(
            f"resume snapshot name says epoch/step {(epoch_expected, step_expected)}, "
            f"but its state contains {(epoch, step)}."
        )
    extra = state["extra_state"]
    if extra.get("best_checkpoint_relative") != "best_checkpoint":
        raise CheckpointError("resume snapshot does not reference its embedded best checkpoint.")
    metadata = _read_metadata(path)
    if not isinstance(extra.get("execution"), dict) or extra.get("execution") != metadata.get("execution"):
        raise CheckpointError("resume snapshot execution metadata disagrees with its model bundle.")
    if not isinstance(extra.get("history"), list):
        raise CheckpointError("resume snapshot has no validated history list.")
    return epoch, step


def write_latest_resume_pointer(root: str | Path, checkpoint_dir: str | Path, epoch: int, step: int) -> None:
    root, checkpoint_dir = Path(root), Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    pointer = root / "latest_checkpoint.json"
    temporary = root / f".latest_checkpoint.json.tmp-{uuid.uuid4().hex}"
    temporary.write_text(
        json.dumps(
            {
                "checkpoint_dir": checkpoint_dir.name,
                "completed_epoch": epoch,
                "global_step": step,
                "resume_boundary": "epoch_end",
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, pointer)


def recover_latest_resume_checkpoint(checkpoint_root: str | Path) -> Path | None:
    """Publish complete abandoned staging bundles and repair a stale pointer.

    Incomplete or invalid staging directories are preserved and ignored. All
    publication uses sibling renames; mounted/network filesystems may not offer
    the same crash-atomic guarantees as a local filesystem.
    """
    root = Path(checkpoint_root)
    if not root.is_dir():
        return None

    staging_prefix = ".resume-epoch-"
    for staging in root.iterdir():
        if not staging.is_dir() or not staging.name.startswith(staging_prefix) or ".tmp-" not in staging.name:
            continue
        final_name = staging.name[1:].split(".tmp-", 1)[0]
        final_path = root / final_name
        if final_path.exists():
            continue
        try:
            _validate_resume_snapshot(staging, expected_name=final_name)
            publish_checkpoint_bundle(staging, final_path, require_training_state=True)
        except (CheckpointError, OSError, ValueError, KeyError, TypeError) as exc:
            # Keep abandoned data for inspection; a fresh staging name is used
            # for subsequent attempts, so it cannot block publication.
            warnings.warn(f"Ignoring incomplete or invalid staging resume bundle {staging}: {exc}", RuntimeWarning)
            continue

    valid: list[tuple[int, int, Path]] = []
    for candidate in root.glob("resume-epoch-*"):
        if not candidate.is_dir():
            continue
        try:
            epoch, step = _validate_resume_snapshot(candidate)
        except (CheckpointError, OSError, ValueError, KeyError, TypeError) as exc:
            warnings.warn(f"Ignoring invalid completed resume checkpoint {candidate}: {exc}", RuntimeWarning)
            continue
        valid.append((epoch, step, candidate))
    if not valid:
        return None

    epoch, step, newest = max(valid, key=lambda item: (item[0], item[1], item[2].name))
    write_latest_resume_pointer(root, newest, epoch, step)
    return newest


def resolve_resume_checkpoint(path: str | Path) -> Path:
    """Resolve an explicit bundle, checkpoint root, or latest-pointer path."""
    supplied = Path(path)
    if supplied.is_file() and supplied.name == "latest_checkpoint.json":
        root = supplied.parent
    elif supplied.is_dir() and (
        (supplied / "latest_checkpoint.json").exists()
        or any(supplied.glob("resume-epoch-*"))
        or any(supplied.glob(".resume-epoch-*.tmp-*"))
    ):
        root = supplied
    else:
        _validate_resume_snapshot(supplied)
        return supplied
    recovered = recover_latest_resume_checkpoint(root)
    if recovered is None:
        raise CheckpointError(f"no complete, valid epoch-boundary resume snapshot exists in {root}.")
    return recovered
