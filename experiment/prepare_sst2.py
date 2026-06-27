"""
3_prepare_sst2.py

Creates PyTorch Dataset and DataLoader for
Character-Aware DistilBERT training.

Author: Your Name
"""

import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

from src.phase5_model import (
    tokenizer,
    char_vocab
)

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

TRAIN_PATH = "data/sst2/sst2_train.csv"
VAL_PATH = "data/sst2/sst2_validation.csv"

MAX_LENGTH = 128
CHAR_MAX_LEN = 20

BATCH_SIZE = 16

# -------------------------------------------------------
# Character Encoding
# -------------------------------------------------------

def build_char_ids(tokens):

    char_ids = []

    for token in tokens:

        if token in ["[CLS]", "[SEP]", "[PAD]"]:

            char_ids.append([0] * CHAR_MAX_LEN)
            continue

        token = token.replace("##", "")

        chars = list(token)

        ids = [
            char_vocab.get(c, 0)
            for c in chars[:CHAR_MAX_LEN]
        ]

        ids += [0] * (CHAR_MAX_LEN - len(ids))

        char_ids.append(ids)

    return char_ids


# -------------------------------------------------------
# Dataset
# -------------------------------------------------------

class SST2Dataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(drop=True)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        sentence = self.df.loc[idx, "sentence"]
        label = int(self.df.loc[idx, "label"])

        encoding = tokenizer(
            sentence,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)

        attention_mask = encoding["attention_mask"].squeeze(0)

        tokens = tokenizer.convert_ids_to_tokens(
            input_ids
        )

        char_ids = build_char_ids(tokens)

        char_ids = torch.tensor(char_ids)

        return {

            "input_ids": input_ids,

            "attention_mask": attention_mask,

            "char_ids": char_ids,

            "labels": torch.tensor(label)

        }


# -------------------------------------------------------
# Load CSV
# -------------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

print("Train Samples :", len(train_df))
print("Validation Samples :", len(val_df))

# -------------------------------------------------------
# Create Dataset
# -------------------------------------------------------

train_dataset = SST2Dataset(train_df)

val_dataset = SST2Dataset(val_df)

# -------------------------------------------------------
# DataLoaders
# -------------------------------------------------------

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True

)

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False

)

# -------------------------------------------------------
# Test
# -------------------------------------------------------

batch = next(iter(train_loader))

print("\nBatch Shapes")

print("Input IDs :", batch["input_ids"].shape)

print("Attention :", batch["attention_mask"].shape)

print("Character :", batch["char_ids"].shape)

print("Labels :", batch["labels"].shape)

print("\nEverything looks good!")

# -------------------------------------------------------
# Export
# -------------------------------------------------------

def get_dataloaders():

    return train_loader, val_loader