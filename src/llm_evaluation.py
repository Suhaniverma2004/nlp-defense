import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from src.sanitization import sanitize

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


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


def evaluate_phase3(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        clean = row["clean"]
        adv = row["adversarial"]
        sanitized = sanitize(adv)

        emb_clean = get_embedding(clean)
        emb_adv = get_embedding(adv)
        emb_sanitized = get_embedding(sanitized)

        sim_adv = cosine_similarity(emb_clean, emb_adv)[0][0]
        sim_sanitized = cosine_similarity(emb_clean, emb_sanitized)[0][0]

        rows.append({
            "type": row["type"],
            "clean": clean,
            "adversarial": adv,
            "sanitized": sanitized,
            "sim_adversarial": float(sim_adv),
            "sim_sanitized": float(sim_sanitized),
            "improvement": float(sim_sanitized - sim_adv)
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv("data/adversarial_prompts.csv")

    results = evaluate_phase3(df)

    summary = results.groupby("type")[
        ["sim_adversarial", "sim_sanitized", "improvement"]
    ].mean().reset_index()

    print("\n=== Phase 3: LLM Robustness Evaluation ===")
    print(summary.to_string(index=False))

    results.to_csv("data/phase3_results.csv", index=False)
    summary.to_csv("data/phase3_summary.csv", index=False)

    print("\nSaved:")
    print(" - data/phase3_results.csv")
    print(" - data/phase3_summary.csv")