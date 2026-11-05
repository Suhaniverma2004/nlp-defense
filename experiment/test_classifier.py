"""Small command-line smoke test for the character-aware classifier."""

import torch

from src.charaware_classifier import CharAwareClassifier
from src.phase5_model import prepare_inputs


def run_smoke_test(
    model=None,
    text="This movie was amazing.",
    tokenizer_instance=None,
    max_length=128,
):
    """Return classifier logits for one text, with optional local test fixtures."""
    if model is None:
        model = CharAwareClassifier()

    input_ids, char_ids, attention_mask = prepare_inputs(
        text,
        max_length=max_length,
        tokenizer_instance=tokenizer_instance,
    )

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = input_ids.device

    with torch.no_grad():
        return model(
            input_ids.to(device),
            char_ids.to(device),
            attention_mask.to(device),
        )


if __name__ == "__main__":
    logits = run_smoke_test()
    print("Output shape:", tuple(logits.shape))
