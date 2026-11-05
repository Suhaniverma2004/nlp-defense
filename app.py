"""Streamlit UI for pretrained-baseline similarity-preservation diagnostics."""

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from src.attack_engine import TRANSFORMATIONS
from src.evaluate_defense import MODEL_NAME, get_embedding


st.set_page_config(
    page_title="Text Similarity Diagnostic",
    layout="wide",
    page_icon="🧪",
)

st.title("Text Similarity Preservation Diagnostic")
st.markdown(
    f"Cosine similarity using the named pretrained baseline `{MODEL_NAME}` "
    "(mean-pooled last hidden states)."
)
st.warning(
    "This is a representation diagnostic, not attack detection. Low similarity "
    "does not prove an attack was detected; high similarity does not prove an input is safe."
)

selected_transformations = st.sidebar.multiselect(
    "Select text transformations",
    list(TRANSFORMATIONS),
    default=list(TRANSFORMATIONS),
)

text = st.text_input("Original text", "I love data science")
run = st.button("Run similarity diagnostic")


def _cosine_similarity(left_text, right_text):
    try:
        left = get_embedding(left_text)
        right = get_embedding(right_text)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load or run the `{MODEL_NAME}` baseline. Ensure the model "
            "is available in the Hugging Face cache or that network access is enabled. "
            f"Details: {exc}"
        ) from exc
    return float(cosine_similarity(left, right)[0][0])


if run:
    rows = []
    try:
        for name in selected_transformations:
            transformed = TRANSFORMATIONS[name](text)
            changed = transformed != text
            rows.append({
                "Transformation": name,
                "Original text": text,
                "Transformed text": transformed,
                "Changed": changed,
                "Cosine similarity": _cosine_similarity(text, transformed) if changed else None,
            })
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    results = pd.DataFrame(rows, columns=[
        "Transformation",
        "Original text",
        "Transformed text",
        "Changed",
        "Cosine similarity",
    ])
    changed_rows = results[results["Changed"]] if not results.empty else results
    unchanged_count = int((~results["Changed"]).sum()) if not results.empty else 0

    st.subheader("Similarity summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Changed variants", int(len(changed_rows)))
    col2.metric("Unchanged variants", unchanged_count)
    mean_similarity = changed_rows["Cosine similarity"].mean() if not changed_rows.empty else None
    col3.metric(
        "Mean cosine similarity (changed only)",
        "—" if mean_similarity is None else f"{mean_similarity:.4f}",
    )

    st.caption(
        "Unchanged variants are reported separately and excluded from the mean. "
        "The mean is the arithmetic average of cosine similarities for changed variants."
    )
    st.dataframe(results, use_container_width=True)
