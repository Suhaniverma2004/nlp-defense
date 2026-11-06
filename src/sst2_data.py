"""Auditable inspection and reproducible SST-2 split preparation."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from src.unicode_policy import validate_text

LABEL_MAPPING = {0: "Negative", 1: "Positive"}
SPLIT_SCHEMA_VERSION = 1
SPLIT_STRATEGY = "deduplicate-exact-text-v1; sklearn-stratified-80-10-10-v1"
REQUIRED_COLUMNS = {"sentence", "label"}


class DatasetValidationError(ValueError):
    """Raised when a local SST-2 CSV violates the documented label contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_labeled_sst2(path: str | Path) -> pd.DataFrame:
    """Read and validate exact SST-2 source strings and binary labels."""
    path = Path(path)
    frame = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise DatasetValidationError(f"{path} is missing columns: {sorted(missing_columns)}.")
    if frame["sentence"].isna().any():
        rows = frame.index[frame["sentence"].isna()].tolist()[:10]
        raise DatasetValidationError(f"{path} has missing sentence values at rows {rows}.")
    if frame["label"].isna().any():
        rows = frame.index[frame["label"].isna()].tolist()[:10]
        raise DatasetValidationError(f"{path} has missing labels at rows {rows}.")

    invalid_labels = sorted(set(frame["label"].tolist()) - set(LABEL_MAPPING))
    if invalid_labels:
        raise DatasetValidationError(f"{path} has labels outside {sorted(LABEL_MAPPING)}: {invalid_labels}.")
    if not pd.api.types.is_integer_dtype(frame["label"].dtype):
        raise DatasetValidationError(f"{path} labels must be integer values 0 or 1.")

    for row_index, text in enumerate(frame["sentence"].tolist()):
        try:
            validate_text(text, sample_id=f"{path.name}:{row_index}")
        except (TypeError, ValueError) as exc:
            raise DatasetValidationError(str(exc)) from exc

    if "label_name" in frame.columns:
        expected_names = frame["label"].map(LABEL_MAPPING)
        mismatches = frame.index[frame["label_name"] != expected_names].tolist()
        if mismatches:
            raise DatasetValidationError(
                f"{path} label_name disagrees with label at rows {mismatches[:10]}."
            )
    return frame


def _frame_report(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    text = frame["sentence"]
    duplicated = frame.duplicated(subset=["sentence"], keep=False)
    conflict_groups = frame.groupby("sentence", sort=False)["label"].nunique()
    conflict_texts = set(conflict_groups[conflict_groups > 1].index)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "missing_by_column": {name: int(count) for name, count in frame.isna().sum().items()},
        "label_distribution": {
            str(label): int(count)
            for label, count in frame["label"].value_counts().sort_index().items()
        },
        "exact_duplicate_text_rows": int(duplicated.sum()),
        "exact_duplicate_text_groups": int(text[duplicated].nunique()),
        "conflicting_label_text_groups": int(len(conflict_texts)),
        "conflicting_label_rows": int(frame["sentence"].isin(conflict_texts).sum()),
    }


def inspect_sst2_sources(train_path: str | Path, validation_path: str | Path) -> dict[str, Any]:
    """Return local file statistics, source hashes, and exact-text overlap."""
    train_path, validation_path = Path(train_path), Path(validation_path)
    train = read_labeled_sst2(train_path)
    validation = read_labeled_sst2(validation_path)
    overlap = set(train["sentence"]) & set(validation["sentence"])
    return {
        "repository_declared_provenance": (
            "experiment/dataset_loader.py identifies these as exports from the "
            "GLUE SST-2 dataset; local hashes have not been independently matched upstream."
        ),
        "license_status": "No dataset license file or license metadata is present in this repository.",
        "software_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "train": _frame_report(train, train_path),
        "original_validation": _frame_report(validation, validation_path),
        "train_validation_exact_text_overlap_groups": len(overlap),
        "overlapping_text_examples": sorted(overlap)[:10],
    }


def _distribution(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["label"].value_counts().sort_index()
    return {str(label): int(counts.get(label, 0)) for label in LABEL_MAPPING}


def prepare_reproducible_splits(
    train_path: str | Path,
    original_validation_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Create deterministic, stratified splits from the train source only.

    Exact duplicate text is deduplicated before splitting. Every row belonging
    to a text with conflicting labels is excluded and recorded, preventing
    ambiguous labels from entering any split. The original validation file is
    inspected and hashed but never copied, modified, or mixed into the splits.
    The output directory must be empty or absent.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    train_path, original_validation_path = Path(train_path), Path(original_validation_path)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"split output directory is not empty: {output_dir}")

    source = read_labeled_sst2(train_path).copy()
    original_validation = read_labeled_sst2(original_validation_path)
    source["source_row_index"] = range(len(source))

    label_counts_per_text = source.groupby("sentence", sort=False)["label"].nunique()
    conflicting_texts = set(label_counts_per_text[label_counts_per_text > 1].index)
    conflict_rows = source[source["sentence"].isin(conflicting_texts)].copy()
    unambiguous = source[~source["sentence"].isin(conflicting_texts)].copy()
    duplicate_mask = unambiguous.duplicated(subset=["sentence"], keep="first")
    duplicate_rows_removed = int(duplicate_mask.sum())
    duplicate_rows = unambiguous.loc[duplicate_mask].copy()
    unique = unambiguous.loc[~duplicate_mask].sort_values("source_row_index").reset_index(drop=True)

    if unique["label"].nunique() != 2:
        raise DatasetValidationError("Both sentiment labels must remain after duplicate resolution.")
    if unique["sentence"].duplicated().any():
        raise AssertionError("duplicate text remained after deterministic deduplication.")

    train_validation, test = train_test_split(
        unique,
        test_size=0.1,
        random_state=seed,
        stratify=unique["label"],
    )
    train, validation = train_test_split(
        train_validation,
        test_size=1 / 9,
        random_state=seed,
        stratify=train_validation["label"],
    )
    splits = {
        "train": train.sort_values("source_row_index").reset_index(drop=True),
        "validation": validation.sort_values("source_row_index").reset_index(drop=True),
        "test": test.sort_values("source_row_index").reset_index(drop=True),
    }

    names = list(splits)
    overlap_pairs = {}
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap_pairs[f"{left_name}__{right_name}"] = len(
                set(splits[left_name]["sentence"]) & set(splits[right_name]["sentence"])
            )
    if any(overlap_pairs.values()):
        raise AssertionError(f"exact-text leakage found between splits: {overlap_pairs}")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_files: dict[str, dict[str, Any]] = {}
    for split_name, split_frame in splits.items():
        path = output_dir / f"{split_name}.csv"
        split_frame.to_csv(path, index=False)
        split_files[split_name] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "rows": int(len(split_frame)),
            "label_distribution": _distribution(split_frame),
        }

    excluded_files: dict[str, dict[str, Any]] = {}
    for name, excluded_frame in (
        ("excluded_ambiguous.csv", conflict_rows),
        ("excluded_same_label_duplicates.csv", duplicate_rows),
    ):
        path = output_dir / name
        excluded_frame.to_csv(path, index=False)
        excluded_files[name] = {
            "sha256": sha256_file(path),
            "rows": int(len(excluded_frame)),
        }

    manifest: dict[str, Any] = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "split_strategy": SPLIT_STRATEGY,
        "software_versions": inspect_sst2_sources(train_path, original_validation_path)["software_versions"],
        "seed": seed,
        "proportions_requested": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "label_mapping": {str(key): value for key, value in LABEL_MAPPING.items()},
        "source": {
            "train_path": str(train_path),
            "train_sha256": sha256_file(train_path),
            "train_rows": int(len(source)),
            "original_validation_path": str(original_validation_path),
            "original_validation_sha256": sha256_file(original_validation_path),
            "original_validation_rows": int(len(original_validation)),
            "original_validation_used_in_split": False,
        },
        "duplicate_resolution": {
            "exact_duplicate_text_groups_in_source": int(
                source.loc[source["sentence"].duplicated(keep=False), "sentence"].nunique()
            ),
            "conflicting_label_text_groups": int(len(conflicting_texts)),
            "conflicting_label_rows_excluded": int(len(conflict_rows)),
            "same_label_duplicate_rows_removed": duplicate_rows_removed,
            "policy": "exclude all conflicting-label text rows; keep first source row for same-label exact duplicates",
            "excluded_files": excluded_files,
        },
        "split_files": split_files,
        "exact_text_overlap_groups": overlap_pairs,
        "historical_use_note": (
            "These splits are held out from Milestone 2.3 runs that train only on train.csv. "
            "The legacy repository trainer consumed the full source training CSV and an ignored "
            "local legacy checkpoint exists; its experiment history is unknown. Therefore these "
            "examples are not claimed to be globally untouched across prior project experiments."
        ),
        "original_validation_note": (
            "Preserved in place, not mixed into these splits, and not used by the Milestone 2.3 trainer."
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_split(output_dir: str | Path, split_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one generated split and verify its manifest hash and row count."""
    if split_name not in {"train", "validation", "test"}:
        raise ValueError("split_name must be train, validation, or test.")
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise DatasetValidationError("unsupported split manifest schema version.")
    entry = manifest["split_files"][split_name]
    path = output_dir / entry["path"]
    if sha256_file(path) != entry["sha256"]:
        raise DatasetValidationError(f"{split_name} CSV hash does not match split manifest.")
    frame = read_labeled_sst2(path)
    if len(frame) != entry["rows"] or _distribution(frame) != entry["label_distribution"]:
        raise DatasetValidationError(f"{split_name} CSV rows or labels disagree with split manifest.")
    if frame["sentence"].duplicated().any():
        raise DatasetValidationError(f"{split_name} contains duplicate text.")
    return frame, manifest
