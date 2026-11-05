from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import chatbot_security, evaluate_defense
from src.attack_generator import generate_transformations


def test_generator_exposes_tokenization_noop_as_an_unchanged_variant():
    generated = generate_transformations(["a normal sentence"])
    tokenization = generated[generated["transformation"] == "tokenization"].iloc[0]

    assert tokenization["original"] == "a normal sentence"
    assert tokenization["transformed"] == "a normal sentence"


def test_chatbot_marks_noops_and_summarizes_only_changed_variants(monkeypatch, capsys):
    original = "simple input"
    generated = pd.DataFrame({
        "transformation": ["spacing", "spacing", "unicode", "no_op"],
        "original": [original] * 4,
        "transformed": [original, "simple  input", "sіmple input", original],
    })
    monkeypatch.setattr(chatbot_security, "generate_transformations", lambda _: generated)

    similarity_values = {"simple  input": 0.8, "sіmple input": 0.6}
    calls = []

    def similarity_fn(model, left, right):
        calls.append((model, left, right))
        return similarity_values[right]

    model = object()
    results, summary = chatbot_security.run_chatbot_security(
        original,
        model=model,
        model_name="fixture/checkpoint-17",
        similarity_fn=similarity_fn,
    )

    unchanged = results[results["transformed_text"] == original]
    assert len(unchanged) == 2
    assert unchanged["changed"].tolist() == [False, False]
    assert unchanged["cosine_similarity"].isna().all()
    assert len(calls) == 2
    assert all(call[0] is model for call in calls)

    spacing = summary.set_index("transformation").loc["spacing"]
    unicode = summary.set_index("transformation").loc["unicode"]
    no_op = summary.set_index("transformation").loc["no_op"]
    assert spacing["changed_variant_count"] == 1
    assert spacing["unchanged_variant_count"] == 1
    assert spacing["mean_cosine_similarity_changed_variants"] == pytest.approx(0.8)
    assert unicode["changed_variant_count"] == 1
    assert unicode["unchanged_variant_count"] == 0
    assert unicode["mean_cosine_similarity_changed_variants"] == pytest.approx(0.6)
    assert no_op["changed_variant_count"] == 0
    assert no_op["unchanged_variant_count"] == 1
    assert pd.isna(no_op["mean_cosine_similarity_changed_variants"])

    forbidden_columns = {
        "TP", "FP", "TN", "FN", "accuracy", "robustness", "status", "PASS", "FAIL"
    }
    assert forbidden_columns.isdisjoint(results.columns)
    assert forbidden_columns.isdisjoint(summary.columns)
    output = capsys.readouterr().out
    for forbidden_label in ("TP:", "FP:", "TN:", "FN:", "Accuracy", "Robustness Score", "PASS", "FAIL"):
        assert forbidden_label not in output
    assert "not attack detection" in output.lower()


def test_chatbot_requires_explicit_model_and_provenance():
    with pytest.raises(ValueError, match="explicitly loaded, trained model"):
        chatbot_security.run_chatbot_security("text")

    with pytest.raises(ValueError, match="checkpoint identifier"):
        chatbot_security.run_chatbot_security("text", model=object())


def test_summary_excludes_unchanged_rows_from_mean_independently():
    rows = pd.DataFrame({
        "transformation": ["format", "format", "format", "noop"],
        "changed": [True, True, False, False],
        "cosine_similarity": [0.2, 0.8, 1.0, 1.0],
    })

    summary = evaluate_defense.summarize_similarity_preservation(
        rows,
        changed_column="changed",
    ).set_index("transformation")

    assert summary.loc["format", "changed_variant_count"] == 2
    assert summary.loc["format", "unchanged_variant_count"] == 1
    assert summary.loc["format", "mean_cosine_similarity_changed_variants"] == pytest.approx(0.5)
    assert summary.loc["noop", "changed_variant_count"] == 0
    assert summary.loc["noop", "unchanged_variant_count"] == 1
    assert pd.isna(summary.loc["noop", "mean_cosine_similarity_changed_variants"])


def test_evaluator_names_raw_and_postprocessing_similarity(monkeypatch):
    vectors = {
        "original": np.array([[1.0, 0.0]]),
        "transformed": np.array([[0.0, 1.0]]),
        "processed": np.array([[1.0, 0.0]]),
    }
    monkeypatch.setattr(evaluate_defense, "get_embedding", lambda text: vectors[text])
    monkeypatch.setattr(evaluate_defense, "process_text", lambda text: "processed")
    inputs = pd.DataFrame({
        "type": ["unicode"],
        "clean": ["original"],
        "adversarial": ["transformed"],
    })

    results = evaluate_defense.evaluate_similarity_preservation(inputs)
    row = results.iloc[0]

    assert bool(row["transformation_changed"])
    assert bool(row["defense_changed"])
    assert row["similarity_before_defense"] == pytest.approx(0.0)
    assert row["similarity_after_defense"] == pytest.approx(1.0)
    assert "status" not in results.columns


def test_evaluator_skips_embeddings_for_unchanged_text(monkeypatch):
    embedding_calls = []
    monkeypatch.setattr(evaluate_defense, "get_embedding", lambda text: embedding_calls.append(text))
    monkeypatch.setattr(evaluate_defense, "process_text", lambda text: text)
    inputs = pd.DataFrame({"type": ["noop"], "clean": ["same"], "adversarial": ["same"]})

    results = evaluate_defense.evaluate_similarity_preservation(inputs)

    assert embedding_calls == []
    assert not bool(results.iloc[0]["transformation_changed"])
    assert pd.isna(results.iloc[0]["similarity_before_defense"])
    assert pd.isna(results.iloc[0]["similarity_after_defense"])


def test_dashboard_names_lazy_baseline_and_handles_missing_assets(monkeypatch):
    import app

    assert app.MODEL_NAME == "distilbert-base-uncased"
    assert evaluate_defense.model is None
    monkeypatch.setattr(app, "get_embedding", lambda _: (_ for _ in ()).throw(OSError("offline cache miss")))

    with pytest.raises(RuntimeError, match="distilbert-base-uncased.*Hugging Face cache"):
        app._cosine_similarity("original", "changed")


def test_dashboard_has_only_similarity_diagnostic_labels():
    source = Path("app.py").read_text(encoding="utf-8").lower()

    assert "similarity-preservation diagnostic" in source
    assert "changed variants" in source
    assert "unchanged variants" in source
    for forbidden in ("true positive", "false positive", "true negative", "false negative",
                      "detection accuracy", "robustness", "pass", "fail"):
        assert forbidden not in source


def test_diagnostic_documentation_states_methodological_limits():
    document = " ".join(
        Path("docs/SIMILARITY_DIAGNOSTICS.md").read_text(encoding="utf-8").lower().split()
    )

    assert "similarity preservation is not equivalent to adversarial attack detection" in document
    assert "low similarity does not prove an attack was detected" in document
    assert "high similarity does not prove an input is safe" in document
    assert "ground-truth labels" in document
    assert "trained classifier" in document
