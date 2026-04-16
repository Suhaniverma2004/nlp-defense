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
from src.composite_defense import is_composite

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="LLM Security Dashboard",
    layout="wide",
    page_icon="🛡️"
)

model = CharAwareModel()
model.eval()

BASE_THRESHOLD = 0.85


# ---------------- UTILS ----------------

def compute_similarity(text1, text2):
    input_ids1, char_ids1, mask1 = prepare_inputs(text1)
    input_ids2, char_ids2, mask2 = prepare_inputs(text2)

    with torch.no_grad():
        out1 = model(input_ids1, char_ids1, mask1)
        out2 = model(input_ids2, char_ids2, mask2)

    emb1 = get_sentence_embedding(out1, mask1)
    emb2 = get_sentence_embedding(out2, mask2)

    return F.cosine_similarity(emb1, emb2).item()

def get_risk_level(score):

    if score >= 0.90:
        return "🟢 LOW"
    elif score >= 0.75:
        return "🟡 MEDIUM"
    else:
        return "🔴 HIGH"


def get_attack_category(name):

    name = name.lower()

    if "prompt" in name:
        return "Prompt Injection"
    elif "obfuscation" in name or "unicode" in name:
        return "Obfuscation Attack"
    elif "whitespace" in name or "token" in name or "emoji" in name:
        return "Tokenization Attack"
    elif "mixed" in name:
        return "Mixed Attack"
    else:
        return "Other"


def status_color(status):
    return "🟢" if "PASS" in status else "🔴"


# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")

selected_attacks = st.sidebar.multiselect(
    "Select Attack Types",
    list(ATTACKS.keys()),
    default=list(ATTACKS.keys())
)

st.sidebar.markdown("---")
st.sidebar.info("🛡️ This system evaluates robustness of LLMs against adversarial inputs.")


# ---------------- HEADER ----------------
st.title("🛡️ LLM Security & Robustness Dashboard")
st.markdown("**Real-time adversarial attack detection and evaluation system**")

st.markdown("---")

# ---------------- INPUT ----------------
text = st.text_input("💬 Enter User Input", "I love data science")

run = st.button("🚀 Run Security Analysis")


# ---------------- MAIN ----------------
if run:

    results = []

    for name, attack_fn in ATTACKS.items():

        if name not in selected_attacks:
            continue

        attacked = attack_fn(text)
        score = compute_similarity(text, attacked)

        threshold = 0.70 if is_composite(attacked) else BASE_THRESHOLD
        status = "PASS" if score >= threshold else "FAIL"

        results.append({
            "Attack": name,
            "Category": get_attack_category(name),
            "Similarity": round(score, 3),
            "Risk": get_risk_level(score),
            "Threshold": threshold,
            "Status": status,
            "Attacked Text": attacked
        })

        results.insert(0, {
         "Attack": "clean",
         "Category": "None",
        "Similarity": 1.0,
        "Threshold": BASE_THRESHOLD,
        "Risk": "🟢 LOW",
        "Status": "PASS",
        "Attacked Text": text
        })


    df = pd.DataFrame(results)

    # ---------------- METRICS CARDS ----------------
    pass_count = sum(df["Status"] == "PASS")
    total = len(df)
    robustness = pass_count / total * 100

    TP = FP = TN = FN = 0

    for _, row in df.iterrows():
        is_attack = row["Attack"] != "clean"
        pred_attack = row["Status"] == "FAIL"

        if is_attack and pred_attack:
            TP += 1
        elif is_attack and not pred_attack:
            FN += 1
        elif not is_attack and not pred_attack:
            TN += 1
        else:
            FP += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)

    col1, col2, col3 = st.columns(3)

    col1.metric("🛡️ Robustness", f"{robustness:.2f}%")
    col2.metric("📊 Accuracy", f"{accuracy:.3f}")
    col3.metric("⚠️ Failures", f"{total - pass_count}")

    st.markdown("---")

    # ---------------- TABLE ----------------
    st.subheader("📊 Attack Analysis")

    styled_df = df.copy()
    styled_df["Status"] = styled_df["Status"].apply(
        lambda x: f"{status_color(x)} {x}"
    )

    st.dataframe(styled_df, use_container_width=True)

    # ---------------- INSIGHTS ----------------
    st.markdown("---")
    st.subheader("🧠 Insights")

    best = df.loc[df["Similarity"].idxmax()]
    worst = df.loc[df["Similarity"].idxmin()]

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"🏆 Best Defense: {best['Attack']} ({best['Similarity']})")

    with col2:
        st.error(f"⚠️ Weakest Defense: {worst['Attack']} ({worst['Similarity']})")

    # ---------------- DETAILS ----------------
    with st.expander("🔍 View Detailed Outputs"):
        for _, row in df.iterrows():
            st.markdown(f"**{row['Attack']} ({row['Category']})**")
            st.code(row["Attacked Text"])
            st.markdown("---")
