import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import torch

from src.charaware_classifier import CharAwareClassifier
from src.phase5_model import prepare_inputs

model = CharAwareClassifier()

text = "This movie was amazing."

input_ids, char_ids = prepare_inputs(text)

attention_mask = (input_ids != 0).long()

with torch.no_grad():

    logits = model(
        input_ids,
        char_ids,
        attention_mask
    )

print("Output shape:", logits.shape)