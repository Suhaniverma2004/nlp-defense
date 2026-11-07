"""Static, offline validation for the Milestone 2.4 Kaggle workflow."""

from __future__ import annotations

import json
import ast
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "milestone_2_4_kaggle_gpu.ipynb"


def _notebook() -> tuple[dict, str, str]:
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in data["cells"]
        if cell["cell_type"] == "markdown"
    )
    code = "\n".join(
        "".join(cell["source"])
        for cell in data["cells"]
        if cell["cell_type"] == "code"
    )
    return data, markdown, code


def test_notebook_is_valid_v4_and_code_cells_compile() -> None:
    data, _, _ = _notebook()
    assert data["nbformat"] == 4
    assert data["nbformat_minor"] >= 4
    assert data["metadata"]["kernelspec"]["name"] == "python3"
    for index, cell in enumerate(data["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{NOTEBOOK}:cell-{index}", "exec")
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_required_workflow_sections_and_explicit_training_gates() -> None:
    _, markdown, code = _notebook()
    for heading in (
        "Environment and GPU detection",
        "Dependency compatibility",
        "Repository setup",
        "Dataset provenance",
        "revision verification",
        "Bounded CUDA smoke",
        "GPU memory and timing",
        "Baseline training",
        "Canonical training",
        "comparison manifest",
        "Artifact export",
    ):
        assert heading.casefold() in markdown.casefold()
    assert "RUN_CUDA_SMOKE_TESTS = False" in code
    assert "RUN_FULL_BASELINE_TRAINING = False" in code
    assert "RUN_FULL_CANONICAL_TRAINING = False" in code
    assert "if RUN_FULL_BASELINE_TRAINING:" in code
    assert "if RUN_FULL_CANONICAL_TRAINING:" in code


def test_notebook_pins_revision_cuda_zero_and_avoids_test_split() -> None:
    _, markdown, code = _notebook()
    assert "12040accade4e8a0f71eabdb258fecc2e7e948be" in markdown + code
    assert "DEVICE_NAME = 'cuda:0'" in code
    assert "torch.cuda.set_device(device)" in code
    assert "DataParallel(" not in code
    assert "DistributedDataParallel(" not in code
    assert "load_split(SPLIT_DIR,'test')" not in code
    assert "evaluate_checkpoint_split(checkpoint,SPLIT_DIR,'validation'" in code
    assert "evaluate_checkpoint_split(checkpoint,SPLIT_DIR,'test'" not in code
    assert "whole-system" in markdown + code


def test_notebook_selects_candidate_profile_and_fails_closed_on_exact_versions() -> None:
    _, markdown, code = _notebook()
    from src.kaggle_runtime import EXPECTED_VERSIONS

    assert EXPECTED_VERSIONS == {
        "python": "3.13.16",
        "torch": "2.11.0+cu130",
        "torch_cuda_runtime": "13.0",
        "transformers": "5.20.2",
        "tokenizers": "0.23.3",
        "numpy": "2.1.3",
        "pandas": "2.3.3",
        "scikit-learn": "1.6.1",
    }
    assert "PROFILE_SELECTION = 'kaggle-t4-candidate'" in code
    assert "validate_candidate_runtime" in code
    assert "resolved_package_inventory" in code
    assert "nvidia_driver_version" in code
    assert "profile_status':'candidate'" in code
    assert "pip install" not in code.lower()


def test_notebook_full_training_requires_context_bound_probe_report() -> None:
    _, markdown, code = _notebook()
    assert "RUN_CUDA_SMOKE_TESTS = False" in code
    assert "VALIDATION_RUN_ID = None" in code
    assert "start_probe_run(REPORT_PATH,context)" in code
    assert "validation_context_fingerprint(context)" in code
    assert "run_reported_probe(REPORT_PATH,VALIDATION_RUN_ID,name,fn)" in code
    assert "complete_probe_run(REPORT_PATH,VALIDATION_RUN_ID)" in code
    assert "CANDIDATE_PROBE_RESULTS" not in code
    assert "require_full_training_gate(profile,REPORT_PATH,VALIDATION_RUN_ID,context)" in code
    assert "require_current_candidate_gate()" in code
    assert "source_tree_sha256" in code
    assert "verify_source_partition('literal [CLS] content'" in code
    assert "RUN_FULL_BASELINE_TRAINING = False" in code
    assert "RUN_FULL_CANONICAL_TRAINING = False" in code
    assert "FP16 autocast" in markdown
    assert "epoch-boundary recovery/resume" in markdown
    assert "held-out test split is not loaded" in markdown
    assert "validation_report.json" in code
    assert "runtime_report['validation_report']" in code
    assert "export_current_validation_report" in code
    assert "nlp-defense-m24-validation-" in code


def test_notebook_uses_kaggle_paths_and_does_not_install_dependencies() -> None:
    _, _, code = _notebook()
    assert "/kaggle/input" in code
    assert "/kaggle/working" in code
    assert "pip install" not in code.lower()
    assert "run(['git','clone'" in code


def _notebook_function(name: str, namespace: dict | None = None):
    data, _, _ = _notebook()
    for index, cell in enumerate(data["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]), filename=f"{NOTEBOOK}:cell-{index}")
        function = next(
            (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name),
            None,
        )
        if function is not None:
            scope = {
                "Path": Path,
                "subprocess": subprocess,
                "os": os,
                "contextmanager": contextmanager,
                "patch": patch,
                "sys": sys,
            }
            scope.update(namespace or {})
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), "exec"), scope)
            return scope[name]
    raise AssertionError(f"Notebook function {name!r} was not found.")


def test_repository_setup_requires_explicit_clone_opt_in(tmp_path: Path) -> None:
    calls = []
    setup = _notebook_function("setup_repository")
    with pytest.raises(RuntimeError, match="ALLOW_REPOSITORY_CLONE=True"):
        setup(
            tmp_path / "input-repo",
            tmp_path / "working-repo",
            "main",
            "a" * 40,
            allow_clone=False,
            run=lambda *args, **kwargs: calls.append((args, kwargs)),
        )
    assert calls == []


def test_repository_setup_uses_attached_source_or_explicit_clone(tmp_path: Path) -> None:
    commit = "a" * 40
    setup = _notebook_function("setup_repository")

    attached = tmp_path / "input-repo"
    (attached / "src").mkdir(parents=True)
    (attached / "src" / "training.py").write_text("# source fixture\n", encoding="utf-8")
    (attached / "MILESTONE_2_4_SOURCE_COMMIT.txt").write_text(commit, encoding="utf-8")

    def no_git_metadata(args, **kwargs):
        raise subprocess.CalledProcessError(1, args)

    root, resolved = setup(
        attached,
        tmp_path / "unused-working-repo",
        "main",
        commit,
        allow_clone=False,
        run=no_git_metadata,
    )
    assert root == attached
    assert resolved == commit

    working = tmp_path / "cloned-repo"
    commands = []

    def fake_git(args, **kwargs):
        commands.append(args)
        if args[0] == "git" and args[1] == "clone":
            working.mkdir()
            (working / "src").mkdir()
            (working / "src" / "training.py").write_text("# cloned fixture\n", encoding="utf-8")
        if args[-2:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(args, 0, stdout=commit + "\n")
        return subprocess.CompletedProcess(args, 0, stdout="")

    root, resolved = setup(
        tmp_path / "missing-input-repo",
        working,
        "main",
        commit,
        allow_clone=True,
        run=fake_git,
    )
    assert root == working
    assert resolved == commit
    assert [command[1] for command in commands[:3]] == ["clone", "-C", "-C"]
    assert commands[1][-1] == commit
    assert commands[2][-1] == commit


def test_training_command_propagates_asset_download_opt_in() -> None:
    values = {
        "sys": sys,
        "SPLIT_DIR": Path("splits"),
        "MODEL_ID": "distilbert-base-uncased",
        "MODEL_REVISION": "a" * 40,
        "TOKENIZER_ID": "distilbert-base-uncased",
        "TOKENIZER_REVISION": "a" * 40,
        "MAX_EXPANDED_LENGTH": 128,
        "MAX_BYTE_SPAN_LENGTH": 512,
        "MICROBATCH_SIZE": 2,
        "GRADIENT_ACCUMULATION_STEPS": 8,
        "SEED": 42,
        "EPOCHS": 3,
        "LEARNING_RATE": 2e-5,
        "WEIGHT_DECAY": 0.01,
        "WARMUP_RATIO": 0.1,
        "GRADIENT_CLIP_NORM": 1.0,
        "REQUESTED_MIXED_PRECISION": "fp16",
        "VALIDATION_INTERVAL_STEPS": 0,
        "NUM_WORKERS": 0,
        "DEVICE_NAME": "cuda:0",
    }
    command_builder = _notebook_function("train_command", values)
    for kind in ("baseline", "canonical"):
        command_builder.__globals__["ALLOW_PINNED_ASSET_DOWNLOAD"] = False
        local_command = command_builder(kind, Path("run"))
        assert "--allow-hub-download" not in local_command
        command_builder.__globals__["ALLOW_PINNED_ASSET_DOWNLOAD"] = True
        download_command = command_builder(kind, Path("run"))
        assert download_command.count("--allow-hub-download") == 1
        assert download_command[download_command.index("--model-kind") + 1] == kind


def test_offline_checkpoint_mode_restores_prior_environment(monkeypatch) -> None:
    offline = _notebook_function("temporary_hf_offline")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    with offline():
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert "TRANSFORMERS_OFFLINE" not in os.environ


def test_smoke_and_full_training_share_authoritative_execution_context() -> None:
    _, _, code = _notebook()
    assert "build_authoritative_execution_context(" in code
    assert "build_smoke_training_config(execution['bindings'],out)" in code
    assert "if asdict(cfg)!=expected_config" in code
    assert "torch.cuda.current_device()!=0" in code
    assert "effective_dtype" in code and "cached_dtype=effective_dtype" in code
    assert "def require_current_candidate_gate():" in code
    assert code.count("require_current_candidate_gate()") >= 3
    assert "RUN_CUDA_SMOKE_TESTS = False" in code
    assert "RUN_FULL_BASELINE_TRAINING = False" in code
    assert "RUN_FULL_CANONICAL_TRAINING = False" in code


SOURCE_COMMIT = "65391a2b0a715bdb67156df4aa1d5725c1edbc76"
REQUIRED_SOURCE_COMMIT_PATHS = {
    "notebooks/milestone_2_4_kaggle_gpu.ipynb",
    "requirements.txt",
    "src/alignment.py",
    "src/byte_encoder.py",
    "src/canonical_model.py",
    "src/checkpointing.py",
    "src/comparison.py",
    "src/evaluation.py",
    "src/kaggle_runtime.py",
    "src/sst2_data.py",
    "src/training.py",
    "experiment/train_sst2.py",
}


def test_notebook_source_pin_matches_commit_with_required_runtime_modules() -> None:
    _, _, code = _notebook()
    assert f"REPOSITORY_COMMIT='{SOURCE_COMMIT}'" in code
    root = NOTEBOOK.parents[1]
    listed = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", SOURCE_COMMIT],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    assert REQUIRED_SOURCE_COMMIT_PATHS <= set(listed)


def test_pinned_source_matches_required_checked_out_apis_without_future_source_changes() -> None:
    root = NOTEBOOK.parents[1]
    source_paths = sorted(
        path for path in REQUIRED_SOURCE_COMMIT_PATHS
        if path.startswith(("src/", "experiment/")) or path == "requirements.txt"
    )
    for relative in source_paths:
        pinned_blob = subprocess.run(
            ["git", "rev-parse", f"{SOURCE_COMMIT}:{relative}"],
            cwd=root, check=True, capture_output=True, text=True,
        ).stdout.strip()
        current_blob = subprocess.run(
            ["git", "rev-parse", f"HEAD:{relative}"],
            cwd=root, check=True, capture_output=True, text=True,
        ).stdout.strip()
        assert current_blob == pinned_blob, relative
    changed = subprocess.run(
        ["git", "diff", "--name-only", SOURCE_COMMIT, "HEAD", "--", "src", "experiment", "requirements.txt"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    assert changed == []


def test_source_setup_rejects_a_commit_marker_that_differs_from_the_pin(tmp_path: Path) -> None:
    setup = _notebook_function("setup_repository")
    attached = tmp_path / "attached-source"
    (attached / "src").mkdir(parents=True)
    (attached / "src" / "training.py").write_text("# source fixture\n", encoding="utf-8")
    (attached / "MILESTONE_2_4_SOURCE_COMMIT.txt").write_text("0" * 40, encoding="utf-8")
    def no_git_metadata(args, **kwargs):
        raise subprocess.CalledProcessError(1, args)

    with pytest.raises(RuntimeError, match="must identify inspected commit"):
        setup(attached, tmp_path / "unused", "main", SOURCE_COMMIT, allow_clone=False, run=no_git_metadata)


def test_source_tree_digest_covers_checked_out_runtime_and_ignores_other_worktrees(tmp_path: Path) -> None:
    import hashlib
    digest = _notebook_function("source_tree_digest", {"hashlib": hashlib})
    root = tmp_path / "checked-out-source"
    for folder in ("src", "experiment", "notebooks"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / "src" / "training.py").write_text("training = 1\n", encoding="utf-8")
    runtime = root / "src" / "kaggle_runtime.py"
    runtime.write_text("runtime = 1\n", encoding="utf-8")
    (root / "experiment" / "train_sst2.py").write_text("entrypoint = 1\n", encoding="utf-8")
    (root / "requirements.txt").write_text("torch==pinned\n", encoding="utf-8")
    (root / "notebooks" / "milestone_2_4_kaggle_gpu.ipynb").write_text("{}\n", encoding="utf-8")
    before = digest(root)
    other_worktree = tmp_path / "unrelated-worktree"
    (other_worktree / "src").mkdir(parents=True)
    (other_worktree / "src" / "kaggle_runtime.py").write_text("runtime = unrelated\n", encoding="utf-8")
    assert digest(root) == before
    runtime.write_text("runtime = 2\n", encoding="utf-8")
    assert digest(root) != before


PINNED_HF_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"


def _fake_hf_snapshot(root: Path, *, repo_id: str = "distilbert-base-uncased", revision: str = PINNED_HF_REVISION, missing: str | None = None, include_weights: bool = True) -> Path:
    cache_repo = "models--" + repo_id.replace("/", "--")
    snapshot = root / cache_repo / "snapshots" / revision
    snapshot.mkdir(parents=True)
    assets = {
        "config.json": b'{"model_type":"distilbert"}',
        "tokenizer.json": b'{"version":"1.0"}',
        "tokenizer_config.json": b'{"tokenizer_class":"DistilBertTokenizerFast"}',
        "vocab.txt": b"[PAD]\n[UNK]\n",
        "special_tokens_map.json": b'{"unk_token":"[UNK]"}',
    }
    if include_weights:
        assets["model.safetensors"] = b"test-weights"
    for name, content in assets.items():
        if name != missing:
            (snapshot / name).write_bytes(content)
    return snapshot


def _run_hf_asset_verifier(snapshot: Path, *, model_id: str = "distilbert-base-uncased", model_revision: str = PINNED_HF_REVISION, tokenizer_id: str = "distilbert-base-uncased", tokenizer_revision: str = PINNED_HF_REVISION, local_files_only: bool = True):
    verifier = _notebook_function("verify_pinned_hf_assets", {"Path": Path})
    calls = []
    def snapshot_download_fn(**kwargs):
        calls.append(kwargs)
        return str(snapshot)
    result = verifier(
        model_id, model_revision, tokenizer_id, tokenizer_revision,
        pinned_revision=PINNED_HF_REVISION,
        local_files_only=local_files_only,
        snapshot_download_fn=snapshot_download_fn,
    )
    return result, calls


def test_pinned_hf_verifier_accepts_exact_repo_revision_and_records_required_asset_hashes(tmp_path: Path) -> None:
    snapshot = _fake_hf_snapshot(tmp_path)
    result, calls = _run_hf_asset_verifier(snapshot)
    assert len(calls) == 2
    assert all(call["repo_id"] == "distilbert-base-uncased" for call in calls)
    assert all(call["revision"] == PINNED_HF_REVISION for call in calls)
    assert all(call["local_files_only"] is True for call in calls)
    assert all({"config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "model.safetensors"} <= set(call["allow_patterns"]) for call in calls)
    assert result["model_revision"] == result["tokenizer_revision"] == PINNED_HF_REVISION
    assert result["model_snapshot_identity"] == result["tokenizer_snapshot_identity"]
    assert result["required_assets"] == ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"]
    assert set(result["required_assets"]) <= set(result["asset_sha256"])
    import hashlib
    assert result["asset_sha256"]["config.json"] == hashlib.sha256((snapshot / "config.json").read_bytes()).hexdigest()
    assert result["asset_sha256"]["tokenizer.json"] == hashlib.sha256((snapshot / "tokenizer.json").read_bytes()).hexdigest()
    assert result["weight_assets"] == ["model.safetensors"]


@pytest.mark.parametrize("missing", ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"])
def test_pinned_hf_verifier_rejects_missing_required_asset(tmp_path: Path, missing: str) -> None:
    snapshot = _fake_hf_snapshot(tmp_path, missing=missing)
    with pytest.raises(RuntimeError, match="missing required assets"):
        _run_hf_asset_verifier(snapshot)


def test_pinned_hf_verifier_rejects_missing_model_weights(tmp_path: Path) -> None:
    snapshot = _fake_hf_snapshot(tmp_path, include_weights=False)
    with pytest.raises(RuntimeError, match="no supported model weights"):
        _run_hf_asset_verifier(snapshot)


def test_pinned_hf_verifier_rejects_wrong_repo_snapshot_identity(tmp_path: Path) -> None:
    snapshot = _fake_hf_snapshot(tmp_path, repo_id="other-repository")
    verifier = _notebook_function("verify_pinned_hf_assets", {"Path": Path})
    with pytest.raises(RuntimeError, match="unexpected repository snapshot identity"):
        verifier("distilbert-base-uncased", PINNED_HF_REVISION, "distilbert-base-uncased", PINNED_HF_REVISION, pinned_revision=PINNED_HF_REVISION, local_files_only=True, snapshot_download_fn=lambda **kwargs: str(snapshot))


@pytest.mark.parametrize(
    ("model_id", "model_revision", "tokenizer_id", "tokenizer_revision", "message"),
    [
        ("another/model", PINNED_HF_REVISION, "distilbert-base-uncased", PINNED_HF_REVISION, "repository IDs"),
        ("distilbert-base-uncased", "0" * 40, "distilbert-base-uncased", "0" * 40, "pinned revision"),
        ("distilbert-base-uncased", PINNED_HF_REVISION, "distilbert-base-uncased", "0" * 40, "must be identical"),
        ("distilbert-base-uncased", "main", "distilbert-base-uncased", "main", "40-character commit SHA"),
    ],
)
def test_pinned_hf_verifier_rejects_inconsistent_or_mutable_requests(tmp_path: Path, model_id: str, model_revision: str, tokenizer_id: str, tokenizer_revision: str, message: str) -> None:
    snapshot = _fake_hf_snapshot(tmp_path)
    verifier = _notebook_function("verify_pinned_hf_assets", {"Path": Path})
    with pytest.raises(RuntimeError, match=message):
        verifier(model_id, model_revision, tokenizer_id, tokenizer_revision, pinned_revision=PINNED_HF_REVISION, local_files_only=True, snapshot_download_fn=lambda **kwargs: str(snapshot))


def test_pinned_hf_verifier_rejects_snapshot_with_wrong_revision(tmp_path: Path) -> None:
    snapshot = _fake_hf_snapshot(tmp_path, revision="0" * 40)
    with pytest.raises(RuntimeError, match="unexpected repository snapshot identity"):
        _run_hf_asset_verifier(snapshot)


def test_notebook_records_verified_assets_and_avoids_private_commit_hash_fields() -> None:
    _, _, code = _notebook()
    assert "from huggingface_hub import snapshot_download" in code
    assert "_commit_hash" not in code
    assert "verified_hf_assets" in code
    assert "asset_verification" in code
    assert "fresh_asset_verification" in code
    assert "local_files_only=not ALLOW_PINNED_ASSET_DOWNLOAD" in code
    assert "runtime_report.update({'model_id':MODEL_ID,'model_revision':asset_verification['model_revision']" in code
    assert "'verified_hf_assets':asset_verification" in code


def test_training_pipeline_keeps_explicit_pinned_from_pretrained_calls() -> None:
    root = NOTEBOOK.parents[1]
    source = (root / "src" / "training.py").read_text(encoding="utf-8")
    assert "PINNED_REPRODUCIBILITY_REVISION = \"12040accade4e8a0f71eabdb258fecc2e7e948be\"" in source
    assert "config.tokenizer_id,\n            revision=config.tokenizer_revision" in source
    assert "config.model_id,\n            revision=config.model_revision" in source


def test_pinned_hf_verifier_fails_closed_when_snapshot_is_unavailable() -> None:
    verifier = _notebook_function("verify_pinned_hf_assets", {"Path": Path})
    def unavailable(**kwargs):
        raise FileNotFoundError("pinned snapshot is not cached")
    with pytest.raises(RuntimeError, match="Could not resolve model snapshot"):
        verifier("distilbert-base-uncased", PINNED_HF_REVISION, "distilbert-base-uncased", PINNED_HF_REVISION, pinned_revision=PINNED_HF_REVISION, local_files_only=True, snapshot_download_fn=unavailable)


def test_supported_huggingface_hub_snapshot_download_api_is_available() -> None:
    import inspect
    from huggingface_hub import snapshot_download
    parameters = inspect.signature(snapshot_download).parameters
    assert {"repo_id", "revision", "local_files_only", "allow_patterns"} <= set(parameters)


def test_notebook_loaders_keep_explicit_pinned_revisions_and_cache_switch() -> None:
    _, _, code = _notebook()
    assert "AutoTokenizer.from_pretrained(TOKENIZER_ID,revision=TOKENIZER_REVISION,use_fast=True,local_files_only=local_only)" in code
    assert "AutoConfig.from_pretrained(MODEL_ID,revision=MODEL_REVISION,local_files_only=local_only)" in code
    assert "MODEL_REVISION==TOKENIZER_REVISION==PINNED_REPRODUCIBILITY_REVISION" in code
