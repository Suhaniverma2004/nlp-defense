from pathlib import Path
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data/sst2/sst2_train.csv"
VAL_PATH = ROOT / "data/sst2/sst2_validation.csv"
OUTPUT_DIR = ROOT / "data/processed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def clean_dataset():
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VAL_PATH)

    required = {"sentence", "label", "idx", "label_name"}

    for name, df in [("train", train), ("validation", validation)]:
        if not required.issubset(df.columns):
            raise ValueError(f"Invalid columns in {name}")

        if df["sentence"].isna().any():
            raise ValueError(f"Missing sentence in {name}")

        if not df["label"].isin([0, 1]).all():
            raise ValueError(f"Invalid labels in {name}")

    counts = train.groupby("sentence")["label"].nunique()
    conflicting = set(counts[counts > 1].index)

    ambiguous = train[
        train["sentence"].isin(conflicting)
    ].copy()

    clean_train = train[
        ~train["sentence"].isin(conflicting)
    ].drop_duplicates(
        subset=["sentence"],
        keep="first"
    ).reset_index(drop=True)

    clean_validation = validation.copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "train": OUTPUT_DIR / "sst2_train_clean.csv",
        "validation": OUTPUT_DIR / "sst2_validation.csv",
        "ambiguous": OUTPUT_DIR / "sst2_ambiguous.csv",
    }

    clean_train.to_csv(outputs["train"], index=False)
    clean_validation.to_csv(outputs["validation"], index=False)
    ambiguous.to_csv(outputs["ambiguous"], index=False)

    assert clean_train["sentence"].is_unique

    overlap = (
        set(clean_train["sentence"])
        & set(clean_validation["sentence"])
    )

    assert not overlap, "Train-validation overlap detected"

    print("DATASET CLEANING COMPLETE")
    print("Original training rows:", len(train))
    print("Ambiguous rows excluded:", len(ambiguous))
    print("Clean training rows:", len(clean_train))
    print("Validation rows:", len(clean_validation))
    print("Train-validation overlap:", len(overlap))

    for name, path in outputs.items():
        print(f"{name}: {path}")
        print(f"SHA256: {sha256(path)}")


if __name__ == "__main__":
    clean_dataset()
