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
