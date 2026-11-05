import torch
import pytest
from transformers import DistilBertConfig

from experiment.test_classifier import run_smoke_test
from src.char_vocab import build_char_vocab
from src.charaware_classifier import CharAwareClassifier
from src.chatbot_security import run_chatbot_security
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


def test_chatbot_integration_uses_attack_dataframe_and_metric_contract():
    sentinel_model = object()

    def local_similarity(model, clean_text, attacked_text):
        assert model is sentinel_model
        return 1.0 if clean_text == attacked_text else 0.4

    results, metrics = run_chatbot_security(
        "simple input",
        model=sentinel_model,
        similarity_fn=local_similarity,
    )

    assert set(("type", "text", "similarity", "status")).issubset(results.columns)
    assert "clean" in results["type"].values

    tp, fp, tn, fn, accuracy, robustness = metrics
    expected_tp = int(((results["type"] != "clean") & (results["status"] == "FAIL")).sum())
    expected_fp = int(((results["type"] == "clean") & (results["status"] == "FAIL")).sum())
    expected_tn = int(((results["type"] == "clean") & (results["status"] == "PASS")).sum())
    expected_fn = int(((results["type"] != "clean") & (results["status"] == "PASS")).sum())

    assert (tp, fp, tn, fn) == (expected_tp, expected_fp, expected_tn, expected_fn)
    assert accuracy == pytest.approx((tp + tn) / len(results))
    assert robustness == pytest.approx(tp / (tp + fn))
