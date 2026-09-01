import numpy as np
import pandas as pd

# =====================================================
# Load Detailed Results
# =====================================================

baseline = pd.read_csv("data/baseline_results.csv")
defense = pd.read_csv("data/defense_results.csv")
phase3 = pd.read_csv("data/phase3_results.csv")


# =====================================================
# Phase 1 : Baseline
# =====================================================

baseline_summary = (
    baseline.groupby("type")["similarity"]
    .agg(
        mean="mean",
        std=lambda x: np.std(x, ddof=0)
    )
    .reset_index()
)

baseline_summary.columns = [
    "attack",
    "baseline_mean",
    "baseline_std"
]


# =====================================================
# Phase 2 : Defense
# =====================================================

defense_summary = (
    defense.groupby("type")["similarity_after"]
    .agg(
        mean="mean",
        std=lambda x: np.std(x, ddof=0)
    )
    .reset_index()
)

defense_summary.columns = [
    "attack",
    "defense_mean",
    "defense_std"
]


# =====================================================
# Phase 3 : Final Sanitized
# =====================================================

phase3_summary = (
    phase3.groupby("type")["sim_sanitized"]
    .agg(
        mean="mean",
        std=lambda x: np.std(x, ddof=0)
    )
    .reset_index()
)

phase3_summary.columns = [
    "attack",
    "final_mean",
    "final_std"
]


# =====================================================
# Merge All Results
# =====================================================

df = baseline_summary.merge(defense_summary, on="attack")
df = df.merge(phase3_summary, on="attack")


# =====================================================
# Round Numeric Values
# =====================================================

numeric_cols = [
    "baseline_mean",
    "baseline_std",
    "defense_mean",
    "defense_std",
    "final_mean",
    "final_std"
]

df[numeric_cols] = df[numeric_cols].round(3)


# =====================================================
# Create IEEE Table II Columns (mean ± std)
# =====================================================

df["Baseline"] = (
    df["baseline_mean"].astype(str)
    + " ± "
    + df["baseline_std"].astype(str)
)

df["Defense"] = (
    df["defense_mean"].astype(str)
    + " ± "
    + df["defense_std"].astype(str)
)

df["Final"] = (
    df["final_mean"].astype(str)
    + " ± "
    + df["final_std"].astype(str)
)


# =====================================================
# Reorder Columns
# =====================================================

paper_table = df[
    [
        "attack",

        # Numeric columns (for plotting)
        "baseline_mean",
        "baseline_std",

        "defense_mean",
        "defense_std",

        "final_mean",
        "final_std",

        # Formatted columns (for paper)
        "Baseline",
        "Defense",
        "Final"
    ]
]


# =====================================================
# Save
# =====================================================

paper_table.to_csv(
    "data/phase4_results.csv",
    index=False
)


# =====================================================
# Print
# =====================================================

print("\n================ Table II Results ================\n")
print(paper_table.to_string(index=False))

print("\nSaved -> data/phase4_results.csv")