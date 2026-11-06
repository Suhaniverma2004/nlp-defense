"""Fail-closed metadata and readiness gates for the Kaggle T4 candidate runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import uuid

KAGGLE_PROFILE_ID = (
    "kaggle-t4x2-py3.13-torch2.11.0+cu130-transformers5.20.2-tokenizers0.23.3-v1"
)
KAGGLE_PROFILE_SELECTION = "kaggle-t4-candidate"
PROFILE_STATUS = "candidate"
EXPECTED_VERSIONS = {
    "python": "3.13.16",
    "torch": "2.11.0+cu130",
    "torch_cuda_runtime": "13.0",
    "transformers": "5.20.2",
    "tokenizers": "0.23.3",
    "numpy": "2.1.3",
    "pandas": "2.3.3",
    "scikit-learn": "1.6.1",
}
EXPECTED_GPU_COUNT = 2
EXPECTED_GPU_NAME = "Tesla T4"
EXPECTED_GPU_CAPABILITY = (7, 5)
MIN_GPU_MEMORY_BYTES = 14 * 2**30
EXPECTED_DEVICE = "cuda:0"
PINNED_MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
REQUIRED_PROBES = frozenset(
    {
        "runtime_profile_validation",
        "cuda_tensor_probe",
        "smoke_split_validation",
        "tokenizer_revision_verified",
        "unicode_emoji_alignment",
        "whitespace_gap_alignment",
        "special_token_offset_handling",
        "baseline_cuda_training_checkpoint_resume_reload",
        "canonical_cuda_training_checkpoint_resume_reload",
        "fp16_autocast_grad_scaler_update",
        "checkpoint_integrity",
        "offline_checkpoint_reload_inference",
        "epoch_boundary_recovery_resume",
    }
)
PROBE_REPORT_SCHEMA = 1


class KaggleRuntimeProfileError(RuntimeError):
    """Raised when the observed runtime or probe evidence is incompatible."""


def validate_candidate_runtime(
    observed_versions: Mapping[str, str],
    gpus: Sequence[Mapping[str, Any]],
    selected_device: str,
    driver_version: str,
    profile_selection: str,
) -> dict[str, Any]:
    """Fail closed unless versions and the two-device T4 inventory match exactly."""
    if profile_selection != KAGGLE_PROFILE_SELECTION:
        raise KaggleRuntimeProfileError(
            f"explicit profile selection must be {KAGGLE_PROFILE_SELECTION!r}; got {profile_selection!r}."
        )
    mismatches = [
        f"{name}: expected {expected!r}, observed {observed_versions.get(name)!r}"
        for name, expected in EXPECTED_VERSIONS.items()
        if observed_versions.get(name) != expected
    ]
    if mismatches:
        raise KaggleRuntimeProfileError(
            "Kaggle candidate runtime mismatch; do not install or replace CUDA PyTorch: "
            + "; ".join(mismatches)
        )
    if selected_device != EXPECTED_DEVICE:
        raise KaggleRuntimeProfileError(
            f"candidate training must use {EXPECTED_DEVICE}; observed {selected_device!r}."
        )
    if len(gpus) != EXPECTED_GPU_COUNT:
        raise KaggleRuntimeProfileError(
            f"expected exactly {EXPECTED_GPU_COUNT} visible T4 GPUs, observed {len(gpus)}."
        )
    for index, gpu in enumerate(gpus):
        if EXPECTED_GPU_NAME.casefold() not in str(gpu.get("name", "")).casefold():
            raise KaggleRuntimeProfileError(f"GPU {index} is not identified as {EXPECTED_GPU_NAME}.")
        capability = tuple(gpu.get("capability", ()))
        if capability != EXPECTED_GPU_CAPABILITY:
            raise KaggleRuntimeProfileError(
                f"GPU {index} capability must be {EXPECTED_GPU_CAPABILITY}, observed {capability}."
            )
        memory_bytes = gpu.get("memory_bytes")
        if isinstance(memory_bytes, bool) or not isinstance(memory_bytes, int) or memory_bytes < MIN_GPU_MEMORY_BYTES:
            raise KaggleRuntimeProfileError(f"GPU {index} reports insufficient or invalid memory capacity.")
    if not isinstance(driver_version, str) or not driver_version.strip():
        raise KaggleRuntimeProfileError("NVIDIA driver version could not be recorded; stop before model execution.")
    return {
        "profile_id": KAGGLE_PROFILE_ID,
        "status": PROFILE_STATUS,
        "versions": dict(observed_versions),
        "driver_version": driver_version.strip(),
        "gpus": [dict(gpu) for gpu in gpus],
        "selected_device": EXPECTED_DEVICE,
        "automatic_multi_gpu": False,
    }


_TRANSIENT_CONTEXT_KEYS = {"path", "output_dir", "artifact_dir", "started_at", "finished_at", "timestamp", "created_at"}


def _stable_context_value(value: Any) -> Any:
    """Drop location/time fields that do not define a validation execution."""
    if isinstance(value, Mapping):
        return {
            str(key): _stable_context_value(item)
            for key, item in value.items()
            if str(key).casefold() not in _TRANSIENT_CONTEXT_KEYS
            and not str(key).casefold().endswith(("_path", "_dir"))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_context_value(item) for item in value]
    return value


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(_stable_context_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise KaggleRuntimeProfileError(f"validation context is not canonical JSON: {exc}") from exc


def validation_context_fingerprint(context: Mapping[str, Any]) -> str:
    """Return deterministic SHA-256 for all material validation inputs."""
    return hashlib.sha256(_canonical(context)).hexdigest()


def _precision_name(value: Any) -> str:
    if value is None:
        return "none"
    return str(value).replace("torch.", "")


def validate_smoke_execution_context(
    *,
    authoritative_shared_config: Mapping[str, Any],
    cached_shared_config: Mapping[str, Any],
    effective_precision: Any,
    cached_effective_precision: Any,
    cached_dtype: Any,
    requested_device: str,
    cached_device: Any,
    active_cuda_index: int,
    smoke_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate notebook caches against the current scalar config and selected device."""
    expected_shared = dict(authoritative_shared_config)
    if dict(cached_shared_config) != expected_shared:
        raise KaggleRuntimeProfileError(
            "cached SHARED training configuration differs from current scalar settings; rerun settings and smoke validation"
        )
    precision = _precision_name(effective_precision)
    if _precision_name(cached_effective_precision) != precision or _precision_name(cached_dtype) != precision:
        raise KaggleRuntimeProfileError(
            "cached effective precision/dtype differs from the currently resolved precision; rerun settings and smoke validation"
        )
    if requested_device != EXPECTED_DEVICE or str(cached_device) != EXPECTED_DEVICE or active_cuda_index != 0:
        raise KaggleRuntimeProfileError(
            "cached device object, requested device and active CUDA device must all match cuda:0"
        )
    overrides = dict(smoke_overrides)
    if overrides != {"epochs": 2, "examples_per_split": 4}:
        raise KaggleRuntimeProfileError(
            "smoke-only overrides must explicitly remain epochs=2 and examples_per_split=4"
        )
    return {
        "shared_training_config": expected_shared,
        "effective_precision": precision,
        "device": EXPECTED_DEVICE,
        "active_cuda_index": 0,
        "smoke_overrides": overrides,
    }


def build_authoritative_execution_context(
    *,
    authoritative_shared_config: Mapping[str, Any],
    cached_shared_config: Mapping[str, Any],
    effective_precision: Any,
    cached_effective_precision: Any,
    cached_dtype: Any,
    requested_device: str,
    cached_device: Any,
    active_cuda_index: int,
    smoke_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the sole validated binding used by smoke execution and its fingerprint."""
    binding = validate_smoke_execution_context(
        authoritative_shared_config=authoritative_shared_config,
        cached_shared_config=cached_shared_config,
        effective_precision=effective_precision,
        cached_effective_precision=cached_effective_precision,
        cached_dtype=cached_dtype,
        requested_device=requested_device,
        cached_device=cached_device,
        active_cuda_index=active_cuda_index,
        smoke_overrides=smoke_overrides,
    )
    if binding["device"] != EXPECTED_DEVICE or binding["active_cuda_index"] != 0:
        raise KaggleRuntimeProfileError("validated smoke execution must be bound to active cuda:0")
    return binding


def build_smoke_training_config(execution_context: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Apply only the declared bounded-epoch and output-location smoke overrides."""
    if not str(output_dir).strip():
        raise KaggleRuntimeProfileError("smoke output directory must be explicitly supplied")
    overrides = execution_context.get("smoke_overrides")
    if overrides != {"epochs": 2, "examples_per_split": 4}:
        raise KaggleRuntimeProfileError("unrecognized or unvalidated smoke-only overrides")
    config = dict(execution_context.get("shared_training_config", {}))
    if not config:
        raise KaggleRuntimeProfileError("fingerprinted shared training configuration is missing")
    config["epochs"] = overrides["epochs"]
    config["output_dir"] = str(output_dir)
    return config


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def initialize_probe_report(path: str | Path, *, preliminary_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Create the durable report before any candidate validation begins."""
    report = {
        "schema": PROBE_REPORT_SCHEMA,
        "report_status": "validation_started",
        "persistence_note": "Atomic replacement is best-effort; guarantees depend on the filesystem and do not cover sudden power loss or mounted/network storage.",
        "preliminary_context": dict(preliminary_context or {}),
        "runs": [],
    }
    _atomic_write_json(Path(path), report)
    return report


def start_probe_run(path: str | Path, context: Mapping[str, Any]) -> str:
    """Append a fresh run with every mandatory probe pending; retain prior evidence."""
    path = Path(path)
    if not path.exists():
        initialize_probe_report(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != PROBE_REPORT_SCHEMA or not isinstance(report.get("runs"), list):
        raise KaggleRuntimeProfileError("probe report is malformed; refusing to authorize training")
    fingerprint = validation_context_fingerprint(context)
    run_id = uuid.uuid4().hex
    report["runs"].append({
        "run_id": run_id,
        "started_at": _now(),
        "status": "running",
        "context": _stable_context_value(context),
        "context_fingerprint": fingerprint,
        "probes": {name: {"status": "pending", "context_fingerprint": fingerprint} for name in sorted(REQUIRED_PROBES)},
    })
    report["report_status"] = "validation_running"
    report["active_run_id"] = run_id
    _atomic_write_json(path, report)
    return run_id


def _find_run(report: dict[str, Any], run_id: str) -> dict[str, Any]:
    for run in report.get("runs", []):
        if run.get("run_id") == run_id:
            return run
    raise KaggleRuntimeProfileError(f"validation run {run_id!r} is absent from the probe report")


def run_reported_probe(path: str | Path, run_id: str, probe_name: str, probe) -> Any:
    """Persist running/pass/failure evidence and always propagate probe exceptions."""
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    run = _find_run(report, run_id)
    if probe_name not in REQUIRED_PROBES:
        raise KaggleRuntimeProfileError(f"unknown mandatory probe {probe_name!r}")
    fingerprint = run["context_fingerprint"]
    previous = run.get("probes", {}).get(probe_name, {})
    if previous.get("status") != "pending":
        raise KaggleRuntimeProfileError(
            f"probe {probe_name!r} already has {previous.get('status')!r} evidence; start a fresh validation run"
        )
    record = {"status": "running", "started_at": _now(), "context_fingerprint": fingerprint}
    run["probes"][probe_name] = record
    run["status"] = "running"
    report["report_status"] = "validation_running"
    _atomic_write_json(path, report)
    try:
        result = probe()
        if result is False:
            raise AssertionError(f"probe {probe_name!r} returned False")
    except BaseException as exc:
        record.update({
            "status": "failed",
            "probe_name": probe_name,
            "finished_at": _now(),
            "exception_type": type(exc).__name__,
            "error": str(exc).replace("\n", " ")[:1200],
            "context_fingerprint": fingerprint,
        })
        run["status"] = "failed"
        report["report_status"] = "validation_failed"
        try:
            _atomic_write_json(path, report)
        except BaseException as persistence_error:
            if hasattr(exc, "add_note"):
                exc.add_note(f"Could not persist probe failure evidence: {persistence_error!r}")
        raise
    record.update({"status": "passed", "probe_name": probe_name, "finished_at": _now(), "context_fingerprint": fingerprint})
    run["status"] = "running"
    _atomic_write_json(path, report)
    return result


def record_probe_failure(path: str | Path, probe_name: str, context: Mapping[str, Any], exc: BaseException) -> None:
    """Persist an early preflight failure even before the complete probe context exists."""
    path = Path(path)
    if not path.exists():
        initialize_probe_report(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = validation_context_fingerprint(context)
    run_id = uuid.uuid4().hex
    now = _now()
    report.setdefault("runs", []).append({
        "run_id": run_id,
        "started_at": now,
        "finished_at": now,
        "status": "failed",
        "context": _stable_context_value(context),
        "context_fingerprint": fingerprint,
        "probes": {name: {"status": "pending", "context_fingerprint": fingerprint} for name in sorted(REQUIRED_PROBES)},
    })
    report["runs"][-1]["probes"].setdefault(probe_name, {}).update({
        "status": "failed",
        "probe_name": probe_name,
        "finished_at": now,
        "exception_type": type(exc).__name__,
        "error": str(exc).replace("\n", " ")[:1200],
        "context_fingerprint": fingerprint,
    })
    report["active_run_id"] = run_id
    report["report_status"] = "validation_failed"
    _atomic_write_json(path, report)


def validate_probe_report(path: str | Path, run_id: str, context: Mapping[str, Any]) -> None:
    """Require complete, successful evidence from the active identical context."""
    path = Path(path)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        run = _find_run(report, run_id)
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise KaggleRuntimeProfileError(f"probe evidence is missing or corrupt: {exc}") from exc
    expected = validation_context_fingerprint(context)
    if report.get("active_run_id") != run_id or run.get("context_fingerprint") != expected:
        raise KaggleRuntimeProfileError("probe evidence is stale or belongs to a different validation context")
    problems = []
    for name in sorted(REQUIRED_PROBES):
        result = run.get("probes", {}).get(name, {})
        if result.get("status") != "passed":
            problems.append(f"{name}={result.get('status', 'missing')}")
        if result.get("context_fingerprint") != expected:
            problems.append(f"{name}=fingerprint-mismatch")
    if run.get("status") != "passed":
        problems.append(f"run={run.get('status', 'incomplete')}")
    if problems:
        raise KaggleRuntimeProfileError("full training is gated; incomplete or failed probes: " + ", ".join(problems))


def complete_probe_run(path: str | Path, run_id: str) -> None:
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    run = _find_run(report, run_id)
    if any(run.get("probes", {}).get(name, {}).get("status") != "passed" for name in REQUIRED_PROBES):
        raise KaggleRuntimeProfileError("cannot complete a probe run with pending or failed probes")
    run["status"] = "passed"
    run["finished_at"] = _now()
    report["report_status"] = "validation_passed"
    _atomic_write_json(path, report)


def require_full_training_gate(profile: Mapping[str, Any], report_path: str | Path, run_id: str, context: Mapping[str, Any]) -> None:
    """Fail closed unless a complete report matches the freshly recomputed context."""
    if profile.get("profile_id") != KAGGLE_PROFILE_ID or profile.get("status") != PROFILE_STATUS:
        raise KaggleRuntimeProfileError("a validated Kaggle T4 candidate profile is required before full training.")
    if profile.get("selected_device") != EXPECTED_DEVICE or profile.get("automatic_multi_gpu") is not False:
        raise KaggleRuntimeProfileError("candidate training must remain single-device on cuda:0.")
    validate_candidate_runtime(
        profile.get("versions", {}),
        profile.get("gpus", []),
        profile.get("selected_device", ""),
        profile.get("driver_version", ""),
        KAGGLE_PROFILE_SELECTION,
    )
    validate_probe_report(report_path, run_id, context)


def build_runtime_provenance(
    *,
    profile: Mapping[str, Any],
    package_inventory: Sequence[str],
    repository_commit: str,
    model_revision: str,
    tokenizer_revision: str,
    dataset_manifest: Mapping[str, Any],
    effective_precision: str,
    training_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the provenance object persisted with Kaggle experiment artifacts."""
    if model_revision != PINNED_MODEL_REVISION or tokenizer_revision != PINNED_MODEL_REVISION:
        raise KaggleRuntimeProfileError("model and tokenizer revisions must equal the approved immutable test revision.")
    if len(repository_commit) != 40 or any(c not in "0123456789abcdef" for c in repository_commit.lower()):
        raise KaggleRuntimeProfileError("repository_commit must be a full 40-character Git SHA.")
    if not package_inventory:
        raise KaggleRuntimeProfileError("resolved package inventory is required.")
    if profile.get("profile_id") != KAGGLE_PROFILE_ID:
        raise KaggleRuntimeProfileError("runtime provenance must name the Kaggle candidate profile.")
    source_hashes = dataset_manifest.get("source", {})
    split_files = dataset_manifest.get("split_files", {})
    required_hashes = [source_hashes.get("train_sha256"), source_hashes.get("original_validation_sha256")]
    required_hashes.extend(
        split_files.get(name, {}).get("sha256") for name in ("train", "validation", "test")
    )
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
        for value in required_hashes
    ):
        raise KaggleRuntimeProfileError("dataset manifest must contain SHA-256 hashes for sources and all split files.")
    serialized_manifest = json.dumps(
        dataset_manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return {
        "runtime_profile": dict(profile),
        "package_inventory": sorted(package_inventory),
        "repository_commit": repository_commit.lower(),
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "dataset_split_manifest": dict(dataset_manifest),
        "dataset_split_manifest_sha256": hashlib.sha256(serialized_manifest).hexdigest(),
        "effective_precision": effective_precision,
        "training_config": dict(training_config),
        "comparison_scope": "whole-system; byte-feature effect is not isolated",
    }
