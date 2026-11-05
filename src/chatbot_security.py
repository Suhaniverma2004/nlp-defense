"""Similarity-preservation diagnostics for generated text transformations.

Cosine similarity is a representation diagnostic, not attack detection. Low
similarity does not prove an attack was detected, and high similarity does
not establish that an input is safe.
"""

import pandas as pd

from src.attack_generator import generate_transformations
from src.evaluate_defense import summarize_similarity_preservation
from src.phase5_model import compute_similarity


def run_chatbot_security(
    original_text,
    model=None,
    model_name=None,
    similarity_fn=None,
):
    """Return per-transformation cosine diagnostics and changed-only summaries.

    A model must be supplied explicitly and must have been loaded from a
    documented, trained checkpoint. No model is initialized by this function.
    Similarities compare the raw original and generated text with preprocessing
    disabled. Identical variants receive no score and are excluded from means.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]:
            Per-transformation rows and changed-only summary rows.
    """
    if model is None:
        raise ValueError(
            "Supply an explicitly loaded, trained model and its checkpoint; "
            "this diagnostic does not create or validate model weights."
        )
    if not model_name or not model_name.strip():
        raise ValueError("Provide a model_name or checkpoint identifier for provenance.")

    if similarity_fn is None:
        similarity_fn = lambda supplied_model, original, transformed: compute_similarity(
            supplied_model,
            original,
            transformed,
            apply_defense=False,
        )

    generated = generate_transformations([original_text])
    rows = []

    for variant in generated.itertuples(index=False):
        changed = variant.transformed != original_text
        score = (
            float(similarity_fn(model, original_text, variant.transformed))
            if changed
            else None
        )
        rows.append({
            "transformation": variant.transformation,
            "original_text": original_text,
            "transformed_text": variant.transformed,
            "changed": changed,
            "model_name": model_name,
            "cosine_similarity": score,
        })

    results = pd.DataFrame(rows)
    summary = summarize_similarity_preservation(
        results,
        group_column="transformation",
        changed_column="changed",
        similarity_column="cosine_similarity",
    )

    print("\nSimilarity-preservation diagnostic (not attack detection)")
    print(f"Embedding model/checkpoint: {model_name}")
    print(summary.to_string(index=False))
    return results, summary
