import torch
import torch.nn as nn

from src.phase5_model import (
    CharAwareModel,
    get_sentence_embedding
)


class CharAwareClassifier(nn.Module):
    """
    Character-Aware DistilBERT classifier for SST-2.
    Uses the Phase V encoder and adds a sentiment
    classification head.
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.encoder = CharAwareModel()

        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Linear(768, num_classes)

    def forward(
        self,
        input_ids,
        char_ids,
        attention_mask
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            char_ids=char_ids,
            attention_mask=attention_mask
        )

        pooled = get_sentence_embedding(
            outputs,
            attention_mask
        )

        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)

        return logits