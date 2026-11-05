import torch
from transformers import DistilBertConfig

from experiment.test_classifier import run_smoke_test
from src.char_vocab import build_char_vocab
from src.charaware_classifier import CharAwareClassifier
from src import phase5_model
from src.phase5_model import prepare_inputs


class LocalTokenizer:
    """Minimal tokenizer fixture with the shape contract used by inference."""

    def __call__(self, text, padding, truncation, max_length, return_tensors):
        token_ids = [101, 2023, 224, 102]
        token_ids = token_ids[:max_length]
        attention = [1] * len(token_ids)
        pad_length = max_length - len(token_ids)
        token_ids.extend([0] * pad_length)
        attention.extend([0] * pad_length)
        return {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention], dtype=torch.long),
        }

    def convert_ids_to_tokens(self, input_ids):
        tokens = ["[CLS]", "this", "é", "[SEP]"]
        return (tokens + ["[PAD]"] * len(input_ids))[:len(input_ids)]


def local_model_config():
    return DistilBertConfig(
        vocab_size=30522,
        max_position_embeddings=32,
        n_layers=1,
        dim=768,
        hidden_dim=128,
        n_heads=12,
    )


def test_prepare_inputs_returns_model_contract_and_uses_unknown_id():
    input_ids, char_ids, attention_mask = prepare_inputs(
        "text",
        max_length=8,
        apply_defense=False,
        tokenizer_instance=LocalTokenizer(),
    )

    assert input_ids.shape == (1, 8)
    assert char_ids.shape == (1, 8, 20)
    assert attention_mask.shape == (1, 8)
    assert char_ids[0, 2, 0].item() == build_char_vocab()["<UNK>"]


def test_classifier_smoke_script_runs_with_local_model_and_tokenizer():
    model = CharAwareClassifier(config=local_model_config())
    model.eval()

    logits = run_smoke_test(
        model=model,
        tokenizer_instance=LocalTokenizer(),
        max_length=8,
    )

    assert logits.shape == (1, 2)


def test_compute_similarity_can_compare_raw_text_without_defense(monkeypatch):
    calls = []

    def local_prepare(text, apply_defense=True):
        calls.append((text, apply_defense))
        vector = [1.0, 0.0] if text == "original" else [0.0, 1.0]
        return (
            torch.tensor([vector]),
            torch.zeros((1, 2, 1), dtype=torch.long),
            torch.ones((1, 2), dtype=torch.long),
        )

    class LocalModel:
        def eval(self):
            return self

        def __call__(self, input_ids, char_ids, attention_mask):
            return input_ids

    monkeypatch.setattr(phase5_model, "prepare_inputs", local_prepare)
    monkeypatch.setattr(phase5_model, "get_sentence_embedding", lambda outputs, mask: outputs)

    similarity = phase5_model.compute_similarity(
        LocalModel(), "original", "transformed", apply_defense=False
    )

    assert similarity == 0.0
    assert calls == [("original", False), ("transformed", False)]
