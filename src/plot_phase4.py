import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================
# IEEE Plot Style
# =====================================================

plt.style.use("seaborn-v0_8-whitegrid")

# =====================================================
# Load Results
# =====================================================

df = pd.read_csv("data/phase4_results.csv")

x = list(range(len(df)))
width = 0.25

plt.figure(figsize=(8, 5))

# =====================================================
# Baseline
# =====================================================

b1 = plt.bar(
    [i - width for i in x],
    df["baseline_mean"],
    width,
    yerr=df["baseline_std"],
    capsize=4,
    color="#4C72B0",
    edgecolor="black",
    linewidth=0.6,
    label="Baseline"
)

# =====================================================
# Defense
# =====================================================

b2 = plt.bar(
    x,
    df["defense_mean"],
    width,
    yerr=df["defense_std"],
    capsize=4,
    color="#DD8452",
    edgecolor="black",
    linewidth=0.6,
    label="Defense"
)

# =====================================================
# Final
# =====================================================

b3 = plt.bar(
    [i + width for i in x],
    df["final_mean"],
    width,
    yerr=df["final_std"],
    capsize=4,
    color="#55A868",
    edgecolor="black",
    linewidth=0.6,
    label="Final"
)

# =====================================================
# Axis Labels
# =====================================================

labels = (
    df["attack"]
    .str.replace("_", " ")
    .str.title()
)

plt.xticks(x, labels, rotation=15, fontsize=10)

plt.xticks(x, labels, fontsize=10)

plt.ylabel("Cosine Similarity", fontsize=11)
plt.xlabel("Attack Type", fontsize=11)

plt.title(
    "Cosine Similarity Across Defense Stages",
    fontsize=12,
    weight="bold"
)

plt.ylim(0, 1.1)

plt.legend(frameon=True)

# =====================================================
# Show Mean Values
# =====================================================

for bars in [b1, b2, b3]:

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.025,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

# =====================================================
# Save
# =====================================================

plt.tight_layout()

plt.savefig(
    "data/phase4_plot.png",
    dpi=600,
    bbox_inches="tight"
)

print("Saved: data/phase4_plot.png")