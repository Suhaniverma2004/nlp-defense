import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def evaluate(df):
    similarities = []
    for _, row in df.iterrows():
        emb_clean = get_embedding(row["clean"])
        emb_adv = get_embedding(row["adversarial"])
        sim = cosine_similarity(emb_clean, emb_adv)[0][0]
        similarities.append(sim)
    df["similarity"] = similarities
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/adversarial_prompts.csv")
    results = evaluate(df)
    print(results.groupby("type")["similarity"].mean())
results.to_csv("data/baseline_results.csv", index=False)
