import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Use a cleaner style (better for papers)
plt.style.use("seaborn-v0_8-whitegrid")

# Load results
df = pd.read_csv("data/phase4_results.csv")

x = range(len(df))
width = 0.25

plt.figure(figsize=(8,5))

# Plot bars
b1 = plt.bar([i - width for i in x], df["baseline"], width,
             label="Baseline (Attack Only)", color="#4C72B0")

b2 = plt.bar(x, df["sanitized"], width,
             label="Sanitized Input", color="#DD8452")

b3 = plt.bar([i + width for i in x], df["final"], width,
             label="Final Robustness", color="#55A868")

# Better attack labels
labels = [
    "Emoji\nInsertion",
    "Emoji\nReplacement",
    "Unicode\nSubstitution",
    "Whitespace\nAttack"
]

plt.xticks(x, labels)

plt.ylabel("Cosine Similarity")
plt.xlabel("Attack Type")

plt.title("Robustness Improvement Across Defense Stages")

plt.ylim(0, 1.1)

plt.legend()

# Add values above bars
for bars in [b1, b2, b3]:
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.02,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

plt.tight_layout()

# Save high‑resolution figure
plt.savefig("data/phase4_plot.png", dpi=300)

print("Saved: data/phase4_plot.png")