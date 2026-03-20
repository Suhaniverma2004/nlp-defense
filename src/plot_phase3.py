import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/phase3_summary.csv")

# Better labels
attack_labels = {
    "emoji_insertion": "Emoji\nInsertion",
    "emoji_replacement": "Emoji\nReplacement",
    "unicode": "Unicode",
    "whitespace": "Whitespace"
}

labels = [attack_labels.get(t, t) for t in df["type"]]

x = range(len(labels))
width = 0.35

plt.figure(figsize=(8,5))

plt.bar([i - width/2 for i in x],
        df["sim_adversarial"],
        width,
        label="Adversarial")

plt.bar([i + width/2 for i in x],
        df["sim_sanitized"],
        width,
        label="Sanitized")

plt.xticks(x, labels)
plt.xlabel("Attack Type")
plt.ylabel("Cosine Similarity")
plt.title("Phase III: LLM Robustness After Sanitization")

plt.ylim(0, 1.1)  # 🔥 IMPORTANT (better scaling)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)  # 🔥 cleaner look

plt.tight_layout()
plt.savefig("data/phase3_plot.png", dpi=200)

print("Saved: data/phase3_plot.png")