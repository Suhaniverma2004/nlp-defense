"""
train_charaware.py

Fine-tunes the Character-Aware DistilBERT model
on the SST-2 sentiment classification dataset.

Outputs:
---------
- Best model (.pt)
- Training history
- Validation accuracy
- Loss curves (CSV)

Author: Your Name
"""

import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from experiment.prepare_sst2 import get_dataloaders
from src.charaware_classifier import CharAwareClassifier

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 3
LR = 2e-5

MODEL_DIR = "models/charaware_sst2"
RESULT_DIR = "results/training"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# --------------------------------------------------
# Data
# --------------------------------------------------

train_loader, val_loader = get_dataloaders()

# --------------------------------------------------
# Model
# --------------------------------------------------

model = CharAwareClassifier()

model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = AdamW(
    model.parameters(),
    lr=LR
)

best_acc = 0

history = []

# --------------------------------------------------
# Validation Function
# --------------------------------------------------

def evaluate():

    model.eval()

    total = 0
    correct = 0
    total_loss = 0

    with torch.no_grad():

        for batch in val_loader:

            input_ids = batch["input_ids"].to(DEVICE)
            char_ids = batch["char_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(
                input_ids,
                char_ids,
                attention_mask
            )

            loss = criterion(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

    accuracy = correct / total

    return accuracy, total_loss / len(val_loader)

# --------------------------------------------------
# Training
# --------------------------------------------------

for epoch in range(EPOCHS):

    print("=" * 60)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print("=" * 60)

    model.train()

    running_loss = 0

    loop = tqdm(train_loader)

    for batch in loop:

        input_ids = batch["input_ids"].to(DEVICE)
        char_ids = batch["char_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        logits = model(
            input_ids,
            char_ids,
            attention_mask
        )

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        loop.set_postfix(loss=loss.item())

    train_loss = running_loss / len(train_loader)

    val_acc, val_loss = evaluate()

    print(f"\nTrain Loss : {train_loss:.4f}")
    print(f"Validation Loss : {val_loss:.4f}")
    print(f"Validation Accuracy : {val_acc:.4f}")

    history.append({

        "epoch": epoch + 1,

        "train_loss": train_loss,

        "val_loss": val_loss,

        "val_accuracy": val_acc

    })

    if val_acc > best_acc:

        best_acc = val_acc

        torch.save(

            model.state_dict(),

            os.path.join(
                MODEL_DIR,
                "charaware_sst2.pt"
            )

        )

        print("\nBest model saved!")

# --------------------------------------------------
# Save History
# --------------------------------------------------

history_df = pd.DataFrame(history)

history_df.to_csv(

    os.path.join(

        RESULT_DIR,

        "training_history.csv"

    ),

    index=False

)

print("\nTraining Complete!")

print(f"\nBest Validation Accuracy : {best_acc:.4f}")