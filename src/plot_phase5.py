import matplotlib.pyplot as plt
import numpy as np

attacks = ["Emoji\nInsertion", "Emoji\nReplacement", "Unicode\nSubstitution", "Whitespace\nAttack"]

baseline = [0.592, 0.678, 0.605, 0.420]
phase4 = [1.000, 0.721, 1.000, 0.753]
phase5 = [0.963, 0.890, 0.889, 0.860]

x = np.arange(len(attacks))
width = 0.25

plt.figure(figsize=(8,5))

b1 = plt.bar(x - width, baseline, width, label="Baseline")
b2 = plt.bar(x, phase4, width, label="Phase IV")
b3 = plt.bar(x + width, phase5, width, label="Phase V")

# value labels
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}",
                 ha='center', fontsize=8)

plt.xticks(x, attacks)
plt.ylim(0, 1.1)
plt.ylabel("Cosine Similarity")
plt.title("Robustness Comparison Across Phases")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("data/phase5_plot.png", dpi=300)
plt.show()