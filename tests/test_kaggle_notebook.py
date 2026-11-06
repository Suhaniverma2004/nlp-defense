"""Static, offline validation for the Milestone 2.4 Kaggle workflow."""

from __future__ import annotations

import json
from pathlib import Path


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
    assert "subprocess.run(['git','clone'" in code
