"""Similarity-preservation diagnostics for text transformations.

The default embedding backbone is the named pretrained model
``distilbert-base-uncased`` with mean pooling over its last hidden states.
These scores describe embedding similarity; they are not attack-detection
predictions or safety judgments.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

from src.sanitization import sanitize
from src.composite_defense import composite_clean, is_composite


MODEL_NAME = "distilbert-base-uncased"

# Loaded only when embedding inference is requested.
tokenizer = None
model = None


def _get_backbone():
    """Load the named pretrained baseline only when diagnostics need it."""
    global tokenizer, model

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if model is None:
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.eval()
    return tokenizer, model


def get_embedding(text: str):
    """Return mean-pooled last-hidden-state embeddings from the named baseline."""
    text_tokenizer, encoder = _get_backbone()
    inputs = text_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    )
    with torch.no_grad():
        outputs = encoder(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()


def process_text(text):
    if is_composite(text):
        return composite_clean(text)
    return sanitize(text)


def evaluate_similarity_preservation(df: pd.DataFrame) -> pd.DataFrame:
    """Compare each original with its raw variant and defense-processed variant.

    ``transformation_changed`` is exact string inequality between ``clean`` and
    ``adversarial``. Similarities are raw cosine similarities in [-1, 1]; this
    function assigns no detection labels or safety outcomes.
    """
    rows = []

    for _, row in df.iterrows():
        original = row["clean"]
        transformed = row["adversarial"]
        processed = process_text(transformed)
        transformation_changed = transformed != original

        if transformation_changed:
            original_embedding = get_embedding(original)
            transformed_embedding = get_embedding(transformed)
            processed_embedding = get_embedding(processed)
            similarity_before_defense = float(
                cosine_similarity(original_embedding, transformed_embedding)[0][0]
            )
            similarity_after_defense = float(
                cosine_similarity(original_embedding, processed_embedding)[0][0]
            )
        else:
            similarity_before_defense = float("nan")
            similarity_after_defense = float("nan")

        rows.append({
            "type": row["type"],
            "clean": original,
            "adversarial": transformed,
            "processed": processed,
            "transformation_changed": transformation_changed,
            "defense_changed": processed != transformed,
            "similarity_before_defense": similarity_before_defense,
            "similarity_after_defense": similarity_after_defense,
        })

    return pd.DataFrame(rows)


def evaluate_with_defense(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible name for the similarity-only evaluation output."""
    return evaluate_similarity_preservation(df)


def summarize_similarity_preservation(
    results: pd.DataFrame,
    group_column="transformation",
    changed_column="transformation_changed",
    similarity_column="cosine_similarity",
) -> pd.DataFrame:
    """Summarize mean similarity over changed variants and count unchanged ones.

    Unchanged rows are excluded from the mean. The mean is undefined (NaN) for
    a group with no changed variants. No threshold or classification is applied.
    """
    required = {group_column, changed_column, similarity_column}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Missing diagnostic columns: {sorted(missing)}")

    rows = []
    for transformation, group in results.groupby(group_column, sort=False, dropna=False):
        changed = group[group[changed_column].astype(bool)]
        scores = changed[similarity_column].dropna()
        rows.append({
            group_column: transformation,
            "changed_variant_count": int(len(changed)),
            "unchanged_variant_count": int((~group[changed_column].astype(bool)).sum()),
            "mean_cosine_similarity_changed_variants": (
                float(scores.mean()) if not scores.empty else float("nan")
            ),
        })

    return pd.DataFrame(rows, columns=[
        group_column,
        "changed_variant_count",
        "unchanged_variant_count",
        "mean_cosine_similarity_changed_variants",
    ])


if __name__ == "__main__":
    df = pd.read_csv("data/adversarial_prompts.csv")
    results = evaluate_similarity_preservation(df)
    summary = summarize_similarity_preservation(
        results,
        group_column="type",
        changed_column="transformation_changed",
        similarity_column="similarity_after_defense",
    )

    print("\nSimilarity-preservation diagnostics")
    print(summary.to_string(index=False))

    results.to_csv("data/defense_results.csv", index=False)
    summary.to_csv("data/defense_summary.csv", index=False)
    print("\nSaved diagnostic results to data/defense_results.csv")
    print("Saved diagnostic summary to data/defense_summary.csv")
