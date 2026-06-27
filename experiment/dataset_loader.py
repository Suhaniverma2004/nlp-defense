"""
load_sst2.py

Step 1 of the downstream evaluation.

Downloads the official SST-2 dataset from GLUE,
inspects it, and saves the train and validation
splits for later adversarial evaluation.

Author: Your Name
"""

import os
import pandas as pd
from datasets import load_dataset

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

OUTPUT_DIR = "data/sst2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------
# Load Official GLUE SST-2
# -------------------------------------------------------

print("=" * 60)
print("Loading SST-2 dataset...")
print("=" * 60)

dataset = load_dataset("glue", "sst2")

train = dataset["train"]
validation = dataset["validation"]

print("\nDataset Loaded Successfully!\n")

# -------------------------------------------------------
# Dataset Statistics
# -------------------------------------------------------

print("Train Samples      :", len(train))
print("Validation Samples :", len(validation))

print("\nFeatures:")
print(train.features)

# -------------------------------------------------------
# Convert to DataFrame
# -------------------------------------------------------

train_df = pd.DataFrame(train)
validation_df = pd.DataFrame(validation)

# -------------------------------------------------------
# Label Mapping
# -------------------------------------------------------

label_map = {
    0: "Negative",
    1: "Positive"
}

train_df["label_name"] = train_df["label"].map(label_map)
validation_df["label_name"] = validation_df["label"].map(label_map)

# -------------------------------------------------------
# Save
# -------------------------------------------------------

train_path = os.path.join(OUTPUT_DIR, "sst2_train.csv")
validation_path = os.path.join(OUTPUT_DIR, "sst2_validation.csv")

train_df.to_csv(train_path, index=False)
validation_df.to_csv(validation_path, index=False)

print("\nSaved:")
print(train_path)
print(validation_path)

# -------------------------------------------------------
# Preview
# -------------------------------------------------------

print("\nTraining Sample\n")
print(train_df.head())

print("\nValidation Sample\n")
print(validation_df.head())

print("\nDone.")


