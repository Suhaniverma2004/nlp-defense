import pandas as pd
import pytest

import experiment.clean_sst2 as cleaner


def create_train_data():
    return pd.DataFrame({
        "sentence": [
            "good movie",
            "good movie",
            "bad movie",
            "ambiguous sentence",
            "ambiguous sentence",
        ],
        "label": [1, 1, 0, 0, 1],
        "idx": [0, 1, 2, 3, 4],
        "label_name": [
            "Positive",
            "Positive",
            "Negative",
            "Negative",
            "Positive",
        ],
    })


def create_validation_data():
    return pd.DataFrame({
        "sentence": ["validation example"],
        "label": [1],
        "idx": [0],
        "label_name": ["Positive"],
    })


def configure_paths(monkeypatch, tmp_path):
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "validation.csv"
    output_dir = tmp_path / "processed"

    create_train_data().to_csv(train_path, index=False)
    create_validation_data().to_csv(val_path, index=False)

    monkeypatch.setattr(cleaner, "TRAIN_PATH", train_path)
    monkeypatch.setattr(cleaner, "VAL_PATH", val_path)
    monkeypatch.setattr(cleaner, "OUTPUT_DIR", output_dir)

    return output_dir


def test_clean_dataset_removes_duplicates_and_conflicts(
    tmp_path, monkeypatch
):
    output_dir = configure_paths(monkeypatch, tmp_path)

    cleaner.clean_dataset()

    clean_train = pd.read_csv(
        output_dir / "sst2_train_clean.csv"
    )
    clean_val = pd.read_csv(
        output_dir / "sst2_validation.csv"
    )
    ambiguous = pd.read_csv(
        output_dir / "sst2_ambiguous.csv"
    )

    assert len(clean_train) == 2
    assert clean_train["sentence"].is_unique

    assert set(clean_train["sentence"]) == {
        "good movie",
        "bad movie",
    }

    assert len(clean_val) == 1
    assert len(ambiguous) == 2

    assert set(ambiguous["sentence"]) == {
        "ambiguous sentence",
    }

    assert not (
        set(clean_train["sentence"])
        & set(clean_val["sentence"])
    )


def test_clean_dataset_rejects_invalid_labels(
    tmp_path, monkeypatch
):
    configure_paths(monkeypatch, tmp_path)

    invalid_train = create_train_data()
    invalid_train.loc[0, "label"] = 2

    invalid_train.to_csv(
        cleaner.TRAIN_PATH,
        index=False,
    )

    with pytest.raises(ValueError, match="Invalid labels"):
        cleaner.clean_dataset()


def test_clean_dataset_rejects_missing_columns(
    tmp_path, monkeypatch
):
    configure_paths(monkeypatch, tmp_path)

    invalid_train = pd.DataFrame({
        "sentence": ["example"],
        "label": [1],
    })

    invalid_train.to_csv(
        cleaner.TRAIN_PATH,
        index=False,
    )

    with pytest.raises(ValueError, match="Invalid columns"):
        cleaner.clean_dataset()
