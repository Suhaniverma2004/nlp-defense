"""Deterministic, local-only fixtures for Milestone 2.3 tests."""

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import DistilBertConfig, DistilBertModel, PreTrainedTokenizerFast


def local_fast_tokenizer() -> PreTrainedTokenizerFast:
    vocabulary = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
        "good": 5,
        "bad": 6,
        "movie": 7,
        "film": 8,
        "nice": 9,
        "boring": 10,
        "plot": 11,
        "story": 12,
        "great": 13,
        "acting": 14,
        "really": 15,
        "fun": 16,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


def tiny_backbone(vocab_size: int, *, seed: int = 33) -> DistilBertModel:
    import torch

    torch.manual_seed(seed)
    config = DistilBertConfig(
        vocab_size=vocab_size,
        max_position_embeddings=40,
        n_layers=1,
        dim=16,
        hidden_dim=32,
        n_heads=2,
        dropout=0.0,
        attention_dropout=0.0,
        seq_classif_dropout=0.0,
    )
    return DistilBertModel(config)


def example_frame():
    import pandas as pd

    return pd.DataFrame(
        [
            {"sentence": "good movie", "label": 1, "idx": 0, "label_name": "Positive"},
            {"sentence": "bad film", "label": 0, "idx": 1, "label_name": "Negative"},
            {"sentence": "nice story", "label": 1, "idx": 2, "label_name": "Positive"},
            {"sentence": "boring plot", "label": 0, "idx": 3, "label_name": "Negative"},
        ]
    )
