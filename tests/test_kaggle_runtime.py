"""Offline tests for the fail-closed Kaggle runtime profile."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.kaggle_runtime import (
    EXPECTED_DEVICE,
    EXPECTED_GPU_CAPABILITY,
    EXPECTED_GPU_NAME,
    EXPECTED_GPU_COUNT,
    EXPECTED_VERSIONS,
    KAGGLE_PROFILE_ID,
    KAGGLE_PROFILE_SELECTION,
    PINNED_MODEL_REVISION,
    REQUIRED_PROBES,
    KaggleRuntimeProfileError,
    build_runtime_provenance,
    require_full_training_gate,
    initialize_probe_report,
    start_probe_run,
    run_reported_probe,
    complete_probe_run,
    validation_context_fingerprint,
    validate_candidate_runtime,
    validate_smoke_execution_context,
    build_authoritative_execution_context,
    build_smoke_training_config,
)


def _valid_observed() -> dict[str, str]:
    return dict(EXPECTED_VERSIONS)


def _valid_gpus() -> list[dict]:
    return [
        {
            "index": index,
            "name": f"{EXPECTED_GPU_NAME} (sm_75)",
            "memory_bytes": int(14.56 * 2**30),
            "capability": list(EXPECTED_GPU_CAPABILITY),
        }
        for index in range(EXPECTED_GPU_COUNT)
    ]


def test_candidate_profile_accepts_exact_kaggle_versions_and_records_cuda0() -> None:
    profile = validate_candidate_runtime(
        _valid_observed(), _valid_gpus(), EXPECTED_DEVICE, "550.90.07", KAGGLE_PROFILE_SELECTION
    )

    assert profile["profile_id"] == KAGGLE_PROFILE_ID
    assert profile["status"] == "candidate"
    assert profile["versions"]["torch"] == "2.11.0+cu130"
    assert profile["versions"]["torch_cuda_runtime"] == "13.0"
    assert profile["selected_device"] == "cuda:0"
    assert profile["automatic_multi_gpu"] is False
    assert len(profile["gpus"]) == 2


@pytest.mark.parametrize(
    ("key", "wrong_value"),
    [
        ("torch", "2.11.0"),  # A missing CUDA build suffix must fail closed.
        ("torch_cuda_runtime", "12.4"),
        ("transformers", "5.2.0"),
        ("tokenizers", "0.23.2"),
        ("numpy", "2.1.4"),
        ("pandas", "2.3.2"),
        ("scikit-learn", "1.6.2"),
        ("python", "3.13.15"),
    ],
)
def test_candidate_profile_rejects_any_package_version_mismatch(key: str, wrong_value: str) -> None:
    versions = _valid_observed()
    versions[key] = wrong_value
    with pytest.raises(KaggleRuntimeProfileError, match="runtime mismatch"):
        validate_candidate_runtime(versions, _valid_gpus(), EXPECTED_DEVICE, "550.90.07", KAGGLE_PROFILE_SELECTION)


def test_candidate_profile_rejects_gpu_inventory_device_and_missing_driver() -> None:
    with pytest.raises(KaggleRuntimeProfileError, match="explicit profile selection"):
        validate_candidate_runtime(_valid_observed(), _valid_gpus(), EXPECTED_DEVICE, "550.90.07", "default")

    with pytest.raises(KaggleRuntimeProfileError, match="exactly 2"):
        validate_candidate_runtime(_valid_observed(), _valid_gpus()[:1], EXPECTED_DEVICE, "550.90.07", KAGGLE_PROFILE_SELECTION)

    bad_gpus = _valid_gpus()
    bad_gpus[1]["capability"] = [8, 0]
    with pytest.raises(KaggleRuntimeProfileError, match="capability"):
        validate_candidate_runtime(_valid_observed(), bad_gpus, EXPECTED_DEVICE, "550.90.07", KAGGLE_PROFILE_SELECTION)

    with pytest.raises(KaggleRuntimeProfileError, match="cuda:0"):
        validate_candidate_runtime(_valid_observed(), _valid_gpus(), "cuda:1", "550.90.07", KAGGLE_PROFILE_SELECTION)

    with pytest.raises(KaggleRuntimeProfileError, match="driver version"):
        validate_candidate_runtime(_valid_observed(), _valid_gpus(), EXPECTED_DEVICE, "", KAGGLE_PROFILE_SELECTION)


def _valid_context() -> dict:
    digest = "a" * 64
    return {
        "runtime_profile": {"profile_id": KAGGLE_PROFILE_ID, "versions": _valid_observed()},
        "package_inventory": ["torch==2.11.0+cu130", "transformers==5.20.2"],
        "device": {"selected": "cuda:0", "driver": "550.90.07", "gpus": _valid_gpus()},
        "repository_commit": "b" * 40,
        "model": {"id": "distilbert-base-uncased", "revision": PINNED_MODEL_REVISION},
        "tokenizer": {"id": "distilbert-base-uncased", "revision": PINNED_MODEL_REVISION},
        "dataset": {"source_sha256": digest, "train_sha256": digest, "validation_sha256": digest},
        "training": {"precision": "float16", "max_expanded_length": 128, "batch_size": 2,
                     "gradient_accumulation_steps": 8, "optimizer": {"name": "AdamW", "lr": 2e-5}},
    }


def _new_run(path, context):
    initialize_probe_report(path, preliminary_context={"profile": KAGGLE_PROFILE_ID})
    return start_probe_run(path, context)


def _pass_all(path, run_id):
    for name in sorted(REQUIRED_PROBES):
        run_reported_probe(path, run_id, name, lambda: True)
    complete_probe_run(path, run_id)


def _gate(path, run_id, context):
    profile = validate_candidate_runtime(_valid_observed(), _valid_gpus(), EXPECTED_DEVICE, "550.90.07", KAGGLE_PROFILE_SELECTION)
    require_full_training_gate(profile, path, run_id, context)


def test_probe_failure_is_durable_and_exception_propagates(tmp_path) -> None:
    path = tmp_path / "validation_report.json"
    context = _valid_context()
    run_id = _new_run(path, context)
    error = ValueError("tokenizer revision unavailable")
    with pytest.raises(ValueError, match="tokenizer revision unavailable"):
        run_reported_probe(path, run_id, "tokenizer_revision_verified", lambda: (_ for _ in ()).throw(error))
    import json
    report = json.loads(path.read_text(encoding="utf-8"))
    record = next(r for r in report["runs"] if r["run_id"] == run_id)["probes"]["tokenizer_revision_verified"]
    assert report["report_status"] == "validation_failed"
    assert record["status"] == "failed"
    assert record["probe_name"] == "tokenizer_revision_verified"
    assert record["exception_type"] == "ValueError"
    assert record["error"] == "tokenizer revision unavailable"
    assert record["finished_at"]
    assert record["context_fingerprint"] == validation_context_fingerprint(context)


def test_probe_status_is_persisted_as_pending_then_running_then_passed(tmp_path) -> None:
    import json
    path = tmp_path / "validation_report.json"
    run_id = _new_run(path, _valid_context())
    before = json.loads(path.read_text(encoding="utf-8"))
    run = next(r for r in before["runs"] if r["run_id"] == run_id)
    assert all(probe["status"] == "pending" for probe in run["probes"].values())

    def inspect_running_state():
        active = json.loads(path.read_text(encoding="utf-8"))
        record = next(r for r in active["runs"] if r["run_id"] == run_id)["probes"]["cuda_tensor_probe"]
        assert record["status"] == "running"
        return True

    run_reported_probe(path, run_id, "cuda_tensor_probe", inspect_running_state)
    after = json.loads(path.read_text(encoding="utf-8"))
    record = next(r for r in after["runs"] if r["run_id"] == run_id)["probes"]["cuda_tensor_probe"]
    assert record["status"] == "passed"
    assert record["probe_name"] == "cuda_tensor_probe"


def test_partial_probe_set_cannot_authorize_training(tmp_path) -> None:
    context = _valid_context(); path = tmp_path / "report.json"; run_id = _new_run(path, context)
    for name in sorted(REQUIRED_PROBES)[:-1]: run_reported_probe(path, run_id, name, lambda: True)
    with pytest.raises(KaggleRuntimeProfileError, match="gated"):
        _gate(path, run_id, context)


def test_failed_probe_cannot_authorize_training(tmp_path) -> None:
    context = _valid_context(); path = tmp_path / "report.json"; run_id = _new_run(path, context)
    for name in sorted(REQUIRED_PROBES):
        if name == "cuda_tensor_probe":
            with pytest.raises(RuntimeError): run_reported_probe(path, run_id, name, lambda: (_ for _ in ()).throw(RuntimeError("CUDA failed")))
        else: run_reported_probe(path, run_id, name, lambda: True)
    with pytest.raises(KaggleRuntimeProfileError, match="cuda_tensor_probe=failed"):
        _gate(path, run_id, context)


@pytest.mark.parametrize("change", [
    lambda c: c["dataset"].update(train_sha256="c" * 64),
    lambda c: c["model"].update(revision="d" * 40),
    lambda c: c["tokenizer"].update(revision="e" * 40),
    lambda c: c["training"].update(learning_rate=1e-5),
    lambda c: c["package_inventory"].append("numpy==2.1.4"),
    lambda c: c["device"].update(driver="551.00"),
    lambda c: c["device"].update(selected="cuda:1"),
    lambda c: c["runtime_profile"]["versions"].update(torch="2.11.0"),
])
def test_changed_validation_context_invalidates_old_evidence(tmp_path, change) -> None:
    import copy
    context = _valid_context(); path = tmp_path / "report.json"; run_id = _new_run(path, context); _pass_all(path, run_id)
    changed = copy.deepcopy(context); change(changed)
    with pytest.raises(KaggleRuntimeProfileError, match="stale|different validation context|cuda:0"):
        _gate(path, run_id, changed)


def test_new_fingerprint_requires_fresh_probe_results_and_preserves_history(tmp_path) -> None:
    import copy, json
    context = _valid_context(); path = tmp_path / "report.json"; old_id = _new_run(path, context); _pass_all(path, old_id)
    changed = copy.deepcopy(context); changed["training"]["epochs"] = 4
    new_id = start_probe_run(path, changed)
    assert new_id != old_id
    with pytest.raises(KaggleRuntimeProfileError, match="stale"):
        _gate(path, old_id, context)
    with pytest.raises(KaggleRuntimeProfileError, match="pending"):
        _gate(path, new_id, changed)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert [r["run_id"] for r in report["runs"]] == [old_id, new_id]
    assert all(r["status"] == "pending" for r in report["runs"][1]["probes"].values())


def test_successful_same_context_probes_authorize_training(tmp_path) -> None:
    context = _valid_context(); path = tmp_path / "report.json"; run_id = _new_run(path, context)
    _pass_all(path, run_id)
    _gate(path, run_id, context)


def test_baseline_and_canonical_notebook_gates_share_context_check() -> None:
    import json
    notebook = Path(__file__).parents[1] / "notebooks" / "milestone_2_4_kaggle_gpu.ipynb"
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    baseline = next(c for c in code if "if RUN_FULL_BASELINE_TRAINING:" in c)
    canonical = next(c for c in code if "if RUN_FULL_CANONICAL_TRAINING:" in c)
    assert "require_current_candidate_gate()" in baseline
    assert "require_current_candidate_gate()" in canonical
    assert "require_full_training_gate" in code[next(i for i,c in enumerate(code) if "def require_current_candidate_gate" in c)]


def test_runtime_provenance_records_inventory_revisions_and_dataset_hashes() -> None:
    profile = validate_candidate_runtime(
        _valid_observed(), _valid_gpus(), EXPECTED_DEVICE, "550.90.07", KAGGLE_PROFILE_SELECTION
    )
    digest = "a" * 64
    manifest = {
        "source": {"train_sha256": digest, "original_validation_sha256": digest},
        "split_files": {
            name: {"sha256": digest} for name in ("train", "validation", "test")
        },
    }
    provenance = build_runtime_provenance(
        profile=profile,
        package_inventory=["torch==2.11.0+cu130", "transformers==5.20.2"],
        repository_commit="b" * 40,
        model_revision=PINNED_MODEL_REVISION,
        tokenizer_revision=PINNED_MODEL_REVISION,
        dataset_manifest=manifest,
        effective_precision="float16",
        training_config={"device": "cuda:0", "batch_size": 2},
    )

    assert provenance["runtime_profile"]["status"] == "candidate"
    assert provenance["package_inventory"] == ["torch==2.11.0+cu130", "transformers==5.20.2"]
    assert provenance["dataset_split_manifest"] == manifest
    assert len(provenance["dataset_split_manifest_sha256"]) == 64
    assert provenance["model_revision"] == provenance["tokenizer_revision"] == PINNED_MODEL_REVISION
    assert provenance["effective_precision"] == "float16"
    assert provenance["training_config"]["device"] == "cuda:0"
    assert "whole-system" in provenance["comparison_scope"]

    malformed = copy.deepcopy(manifest)
    malformed["split_files"]["validation"]["sha256"] = "not-a-hash"
    with pytest.raises(KaggleRuntimeProfileError, match="SHA-256"):
        build_runtime_provenance(
            profile=profile,
            package_inventory=["torch==2.11.0+cu130"],
            repository_commit="b" * 40,
            model_revision=PINNED_MODEL_REVISION,
            tokenizer_revision=PINNED_MODEL_REVISION,
            dataset_manifest=malformed,
            effective_precision="float16",
            training_config={},
        )


def test_fingerprint_is_canonical_and_excludes_transient_paths_and_times() -> None:
    left = _valid_context() | {"artifact_dir": "/kaggle/working/one", "started_at": "2026-01-01T00:00:00Z"}
    right = _valid_context() | {"artifact_dir": "/tmp/another", "started_at": "2027-01-01T00:00:00Z"}
    assert validation_context_fingerprint(left) == validation_context_fingerprint(right)
    assert validation_context_fingerprint({"b": 2, "a": 1}) == validation_context_fingerprint({"a": 1, "b": 2})


def _execution_kwargs():
    shared = {
        "model_id": "distilbert-base-uncased",
        "model_revision": PINNED_MODEL_REVISION,
        "tokenizer_id": "distilbert-base-uncased",
        "tokenizer_revision": PINNED_MODEL_REVISION,
        "max_expanded_length": 128,
        "max_byte_span_length": 512,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "seed": 42,
        "epochs": 3,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "gradient_clip_norm": 1.0,
        "mixed_precision": "fp16",
        "validation_interval_steps": 0,
        "num_workers": 0,
        "device": "cuda:0",
    }
    return {
        "authoritative_shared_config": shared,
        "cached_shared_config": copy.deepcopy(shared),
        "effective_precision": "float16",
        "cached_effective_precision": "float16",
        "cached_dtype": "torch.float16",
        "requested_device": "cuda:0",
        "cached_device": "cuda:0",
        "active_cuda_index": 0,
        "smoke_overrides": {"epochs": 2, "examples_per_split": 4},
    }


def test_matching_execution_context_is_accepted() -> None:
    binding = build_authoritative_execution_context(**_execution_kwargs())
    assert binding["device"] == "cuda:0"
    assert binding["effective_precision"] == "float16"
    assert binding["smoke_overrides"] == {"epochs": 2, "examples_per_split": 4}


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("cached_shared_config", {"learning_rate": 1e-4}),
        ("cached_effective_precision", "none"),
        ("cached_dtype", "torch.bfloat16"),
        ("cached_device", "cuda:1"),
        ("active_cuda_index", 1),
    ],
)
def test_mutated_cached_execution_values_are_rejected(key, value) -> None:
    kwargs = _execution_kwargs()
    if key == "cached_shared_config":
        kwargs[key].update(value)
    else:
        kwargs[key] = value
    with pytest.raises(KaggleRuntimeProfileError, match="cached|active CUDA"):
        validate_smoke_execution_context(**kwargs)


def test_smoke_configs_for_both_model_kinds_match_fingerprint_except_allowed_overrides(tmp_path) -> None:
    binding = build_authoritative_execution_context(**_execution_kwargs())
    baseline = build_smoke_training_config(binding, tmp_path / "baseline")
    canonical = build_smoke_training_config(binding, tmp_path / "canonical")
    assert baseline["epochs"] == canonical["epochs"] == 2
    assert baseline["output_dir"] != canonical["output_dir"]
    baseline.pop("output_dir"); canonical.pop("output_dir")
    assert baseline == canonical
    for key, value in binding["shared_training_config"].items():
        if key != "epochs":
            assert baseline[key] == value


@pytest.mark.parametrize(
    "overrides",
    [
        {"epochs": 1, "examples_per_split": 4},
        {"epochs": 2, "examples_per_split": 8},
        {"epochs": 2, "examples_per_split": 4, "learning_rate": 1e-3},
    ],
)
def test_unrepresented_smoke_overrides_are_rejected(overrides, tmp_path) -> None:
    binding = build_authoritative_execution_context(**_execution_kwargs())
    binding["smoke_overrides"] = overrides
    with pytest.raises(KaggleRuntimeProfileError, match="smoke-only overrides|unvalidated"):
        build_smoke_training_config(binding, tmp_path / "smoke")



def test_authoritative_builder_rejects_scalar_change_before_probe_execution() -> None:
    kwargs = _execution_kwargs()
    kwargs["authoritative_shared_config"] = copy.deepcopy(kwargs["authoritative_shared_config"])
    kwargs["authoritative_shared_config"]["learning_rate"] = 1e-4
    with pytest.raises(KaggleRuntimeProfileError, match="cached SHARED"):
        build_authoritative_execution_context(**kwargs)


def test_smoke_config_is_derived_from_authoritative_builder() -> None:
    kwargs = _execution_kwargs()
    binding = build_authoritative_execution_context(**kwargs)
    baseline = build_smoke_training_config(binding, "baseline-smoke")
    canonical = build_smoke_training_config(binding, "canonical-smoke")
    baseline.pop("output_dir"); canonical.pop("output_dir")
    assert baseline == canonical
    assert baseline["epochs"] == 2
    assert baseline["learning_rate"] == kwargs["authoritative_shared_config"]["learning_rate"]
    assert baseline["device"] == "cuda:0"


def test_mutated_torch_device_object_is_rejected() -> None:
    torch = pytest.importorskip("torch")
    kwargs = _execution_kwargs()
    kwargs["cached_device"] = torch.device("cuda:1")
    with pytest.raises(KaggleRuntimeProfileError, match="cached device object"):
        build_authoritative_execution_context(**kwargs)


def test_matching_torch_device_object_is_accepted() -> None:
    torch = pytest.importorskip("torch")
    kwargs = _execution_kwargs()
    kwargs["cached_device"] = torch.device("cuda:0")
    binding = build_authoritative_execution_context(**kwargs)
    assert binding["device"] == "cuda:0"
    assert binding["active_cuda_index"] == 0
