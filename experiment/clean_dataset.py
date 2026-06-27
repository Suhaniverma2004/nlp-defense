"""
2_clean_accuracy.py

Evaluates the official DistilBERT SST-2 classifier on the
clean SST-2 validation dataset.

Outputs:
---------
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- CSV of predictions

Author: Your Name
"""

import os
import pandas as pd
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

DATA_PATH = "data/sst2/sst2_validation.csv"

OUTPUT_DIR = "results/sst2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------
# Load Validation Dataset
# -------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("Loaded Validation Dataset")
print("=" * 60)

print(df.head())

# -------------------------------------------------------
# Load Official SST-2 Classifier
# -------------------------------------------------------

print("\nLoading DistilBERT SST-2 classifier...\n")

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True
)

print("Model Loaded Successfully\n")

# -------------------------------------------------------
# Run Predictions
# -------------------------------------------------------

predictions = []

for sentence in df["sentence"]:

    result = classifier(sentence)[0]

    pred = 1 if result["label"] == "POSITIVE" else 0

    predictions.append(pred)

df["prediction"] = predictions

# -------------------------------------------------------
# Metrics
# -------------------------------------------------------

y_true = df["label"]
y_pred = df["prediction"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

cm = confusion_matrix(y_true, y_pred)

print("=" * 60)
print("BASELINE RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix\n")
print(cm)

print("\nClassification Report\n")
print(classification_report(y_true, y_pred))

# -------------------------------------------------------
# Save Results
# -------------------------------------------------------

prediction_file = os.path.join(
    OUTPUT_DIR,
    "baseline_predictions.csv"
)

metrics_file = os.path.join(
    OUTPUT_DIR,
    "baseline_metrics.txt"
)

df.to_csv(prediction_file, index=False)

with open(metrics_file, "w") as f:

    f.write(f"Accuracy : {accuracy:.6f}\n")
    f.write(f"Precision: {precision:.6f}\n")
    f.write(f"Recall   : {recall:.6f}\n")
    f.write(f"F1 Score : {f1:.6f}\n\n")

    f.write("Confusion Matrix\n")
    f.write(str(cm))
    f.write("\n\n")

    f.write(classification_report(y_true, y_pred))

print("\nSaved:")
print(prediction_file)
print(metrics_file)

print("\nDone.")