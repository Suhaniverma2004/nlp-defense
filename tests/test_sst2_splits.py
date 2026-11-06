import hashlib

import pandas as pd
import pytest

from src.sst2_data import (
    DatasetValidationError,
    inspect_sst2_sources,
    load_split,
    prepare_reproducible_splits,
    read_labeled_sst2,
)


def make_frame(per_class=20):
    rows = []
    for label in (0, 1):
        label_name = "Negative" if label == 0 else "Positive"
        for index in range(per_class):
            rows.append(
                {
                    "sentence": f"unique {label_name.lower()} example {index}",
                    "label": label,
                    "idx": len(rows),
                    "label_name": label_name,
                }
            )
    rows.extend(
        [
            {"sentence": "unique negative example 0", "label": 0, "idx": 100, "label_name": "Negative"},
            {"sentence": "ambiguous exact text", "label": 0, "idx": 101, "label_name": "Negative"},
            {"sentence": "ambiguous exact text", "label": 1, "idx": 102, "label_name": "Positive"},
        ]
    )
    return pd.DataFrame(rows)


def test_split_is_reproducible_stratified_and_has_no_duplicate_leakage(tmp_path):
    train_path = tmp_path / "source_train.csv"
    original_validation_path = tmp_path / "source_validation.csv"
    train = make_frame()
    original_validation = pd.DataFrame(
        [
            {"sentence": "official dev negative", "label": 0, "idx": 0, "label_name": "Negative"},
            {"sentence": "official dev positive", "label": 1, "idx": 1, "label_name": "Positive"},
        ]
    )
    train.to_csv(train_path, index=False)
    original_validation.to_csv(original_validation_path, index=False)
    validation_bytes_before = original_validation_path.read_bytes()

    first = prepare_reproducible_splits(train_path, original_validation_path, tmp_path / "split-a", seed=42)
    second = prepare_reproducible_splits(train_path, original_validation_path, tmp_path / "split-b", seed=42)

    assert first == second
    assert first["source"]["original_validation_used_in_split"] is False
    assert first["duplicate_resolution"]["conflicting_label_text_groups"] == 1
    assert first["duplicate_resolution"]["conflicting_label_rows_excluded"] == 2
    assert first["duplicate_resolution"]["same_label_duplicate_rows_removed"] == 1
    assert first["exact_text_overlap_groups"] == {
        "train__validation": 0,
        "train__test": 0,
        "validation__test": 0,
    }
    assert original_validation_path.read_bytes() == validation_bytes_before
    assert hashlib.sha256(validation_bytes_before).hexdigest() == first["source"]["original_validation_sha256"]

    expected_counts = {"train": (32, {"0": 16, "1": 16}), "validation": (4, {"0": 2, "1": 2}), "test": (4, {"0": 2, "1": 2})}
    frames = {}
    for name, (count, labels) in expected_counts.items():
        frame, manifest = load_split(tmp_path / "split-a", name)
        frames[name] = frame
        assert len(frame) == count
        assert frame.label.value_counts().sort_index().to_dict() == {int(k): v for k, v in labels.items()}
        assert manifest == first
        assert (tmp_path / "split-a" / f"{name}.csv").read_bytes() == (
            tmp_path / "split-b" / f"{name}.csv"
        ).read_bytes()
    assert not (set(frames["train"].sentence) & set(frames["validation"].sentence))
    assert not (set(frames["train"].sentence) & set(frames["test"].sentence))
    assert not (set(frames["validation"].sentence) & set(frames["test"].sentence))


def test_split_refuses_to_overwrite_nonempty_output(tmp_path):
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    make_frame().to_csv(train_path, index=False)
    make_frame(2).to_csv(validation_path, index=False)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "preserve.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        prepare_reproducible_splits(train_path, validation_path, output)
    assert (output / "preserve.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda frame: frame.assign(sentence=frame.sentence.mask(frame.index == 0)), "missing sentence"),
        (lambda frame: frame.assign(label=2), "labels outside"),
        (lambda frame: frame.assign(label_name="Wrong"), "label_name disagrees"),
    ],
)
def test_dataset_validation_rejects_missing_or_inconsistent_labels(tmp_path, mutation, message):
    path = tmp_path / "invalid.csv"
    frame = mutation(make_frame(2))
    frame.to_csv(path, index=False)
    with pytest.raises(DatasetValidationError, match=message):
        read_labeled_sst2(path)


def test_source_inspection_reports_duplicate_overlap_and_hashes(tmp_path):
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    train = make_frame()
    validation = pd.DataFrame(
        [{"sentence": "unique negative example 0", "label": 0, "idx": 0, "label_name": "Negative"}]
    )
    train.to_csv(train_path, index=False)
    validation.to_csv(validation_path, index=False)

    report = inspect_sst2_sources(train_path, validation_path)

    assert report["train"]["rows"] == len(train)
    assert report["train"]["sha256"] == hashlib.sha256(train_path.read_bytes()).hexdigest()
    assert report["train"]["exact_duplicate_text_groups"] == 2
    assert report["train"]["conflicting_label_text_groups"] == 1
    assert report["train_validation_exact_text_overlap_groups"] == 1
