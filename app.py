import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F

from src.phase5_model import (
    CharAwareModel,
    prepare_inputs,
    get_sentence_embedding
)
from src.attack_engine import ATTACKS

model = CharAwareModel()
model.eval()

THRESHOLD = 0.85


def compute_similarity(text1, text2):
    input_ids1, char_ids1 = prepare_inputs(text1)
    input_ids2, char_ids2 = prepare_inputs(text2)

    attention_mask1 = (input_ids1 != 0).long()
    attention_mask2 = (input_ids2 != 0).long()

    with torch.no_grad():
        out1 = model(input_ids1, char_ids1, attention_mask1)
        out2 = model(input_ids2, char_ids2, attention_mask2)

    emb1 = get_sentence_embedding(out1, attention_mask1)
    emb2 = get_sentence_embedding(out2, attention_mask2)

    return F.cosine_similarity(emb1, emb2).item()


# ---------------- UI ----------------
st.set_page_config(page_title="Phase 5 Dashboard", layout="wide")

st.title("Adversarial Robustness Dashboard")

text = st.text_input("Enter original text:", "I love data science")

if st.button("Run All Attacks"):

    results = []

    for name, attack_fn in ATTACKS.items():
        attacked = attack_fn(text)

        score = compute_similarity(text, attacked)

        status = "✅ PASS" if score >= THRESHOLD else "❌ FAIL"

        results.append({
            "Attack": name,
            "Original": text,
            "Attacked": attacked,
            "Similarity": round(score, 3),
            "Status": status
        })

    df = pd.DataFrame(results)

    st.subheader("📊 Attack Results")
    st.dataframe(df, use_container_width=True)

    # -------- Robustness Score --------
    pass_count = sum(df["Status"] == "✅ PASS")
    total = len(df)
    robustness = pass_count / total * 100

    st.subheader("💯 Robustness Score")
    st.write(f"{robustness:.2f}% ({pass_count}/{total} attacks passed)")

    # -------- Insights --------
    best = df.loc[df["Similarity"].idxmax()]
    worst = df.loc[df["Similarity"].idxmin()]

    st.subheader("🏆 Insights")
    st.write(f"Best handled attack: **{best['Attack']}** ({best['Similarity']})")
    st.write(f"Worst handled attack: **{worst['Attack']}** ({worst['Similarity']})")