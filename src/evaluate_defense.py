import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

from src.sanitization import sanitize
from src.composite_defense import composite_clean, is_composite  # 🔥 NEW

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

THRESHOLD = 0.85  # base threshold


# ---------------- EMBEDDING ----------------
def get_embedding(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()


# ---------------- 🔥 DEFENSE PIPELINE ----------------
def process_text(text):
    if is_composite(text):
        return composite_clean(text)
    return sanitize(text)


# ---------------- EVALUATION ----------------
def evaluate_with_defense(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():

        # 🔥 Apply improved defense
        processed = process_text(row["adversarial"])

        emb_clean = get_embedding(row["clean"])
        emb_processed = get_embedding(processed)

        similarity = cosine_similarity(emb_clean, emb_processed)[0][0]

        # 🔥 Adaptive threshold
        threshold = 0.75 if is_composite(row["adversarial"]) else THRESHOLD
        label = "PASS" if similarity >= threshold else "FAIL"

        rows.append({
            "type": row["type"],
            "clean": row["clean"],
            "adversarial": row["adversarial"],
            "processed": processed,
            "similarity_after": float(similarity),
            "status": label
        })

    return pd.DataFrame(rows)


# ---------------- 🔥 METRICS ----------------
def compute_metrics(df):

    TP = FP = TN = FN = 0

    for _, row in df.iterrows():

        is_attack = row["type"] != "clean"
        pred_attack = row["status"] == "FAIL"

        if is_attack and pred_attack:
            TP += 1
        elif is_attack and not pred_attack:
            FN += 1
        elif not is_attack and not pred_attack:
            TN += 1
        else:
            FP += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    robustness = TP / (TP + FN + 1e-6)

    return TP, FP, TN, FN, accuracy, robustness


# ---------------- MAIN ----------------
if __name__ == "__main__":

    df = pd.read_csv("data/adversarial_prompts.csv")

    results = evaluate_with_defense(df)

    summary = results.groupby("type")["similarity_after"].mean().reset_index()

    # 🔥 Compute metrics
    TP, FP, TN, FN, acc, rob = compute_metrics(results)

    print("\n=== Phase 2: After Defense (Enhanced) ===")
    print(summary.to_string(index=False))

    print("\n📊 Evaluation Metrics")
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Robustness Score: {rob:.3f}")

    results.to_csv("data/defense_results.csv", index=False)
    summary.to_csv("data/defense_summary.csv", index=False)

    print("\nSaved: data/defense_results.csv and data/defense_summary.csv")