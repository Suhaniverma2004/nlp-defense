import pandas as pd

# Load datasets
phase1 = pd.read_csv("data/baseline_results.csv")
phase2 = pd.read_csv("data/comparison.csv")
phase3 = pd.read_csv("data/phase3_summary.csv")

# ---------------- Phase 1 ----------------
phase1 = phase1[["type", "similarity"]]
phase1 = phase1.rename(columns={
    "type": "attack",
    "similarity": "baseline"
})

# average similarity per attack
phase1 = phase1.groupby("attack", as_index=False).mean()


# ---------------- Phase 2 ----------------
phase2 = phase2[["type", "similarity_after"]]
phase2 = phase2.rename(columns={
    "type": "attack",
    "similarity_after": "sanitized"
})


# ---------------- Phase 3 ----------------
phase3 = phase3[["type", "sim_sanitized"]]
phase3 = phase3.rename(columns={
    "type": "attack",
    "sim_sanitized": "final"
})


# ---------------- Merge All ----------------
df = phase1.merge(phase2, on="attack")
df = df.merge(phase3, on="attack")

# Save results
df.to_csv("data/phase4_results.csv", index=False)

print("\nPhase 4 Results:")
print(df)
print("\nSaved -> data/phase4_results.csv")