"""Command-line chatbot security evaluation helpers."""

import pandas as pd

from src.attack_generator import generate_attacks
from src.evaluate_defense import compute_metrics
from src.phase5_model import CharAwareModel, compute_similarity


THRESHOLD = 0.85


def classify(similarity):
    return "PASS" if similarity >= THRESHOLD else "FAIL"


def run_chatbot_security(user_input, model=None, similarity_fn=None):
    """Evaluate generated variants using the shared model and metric contracts.

    ``model`` and ``similarity_fn`` can be supplied by callers and tests. The
    pretrained model and tokenizer are loaded only when real inference runs.
    Returns the per-attack results and ``compute_metrics`` summary.
    """
    if similarity_fn is None:
        similarity_fn = compute_similarity
        if model is None:
            model = CharAwareModel()

    attacks = generate_attacks([user_input])
    rows = []

    print("\nCHATBOT SECURITY SYSTEM\n")

    for attack in attacks.itertuples(index=False):
        similarity = similarity_fn(model, user_input, attack.adversarial)
        status = classify(similarity)
        rows.append({
            "type": attack.type,
            "text": attack.adversarial,
            "similarity": float(similarity),
            "status": status,
        })

        print(f"[{attack.type.upper()}]")
        print(f"Text: {attack.adversarial}")
        print(f"Similarity: {similarity:.3f} -> {status}")
        print("-" * 50)

    results = pd.DataFrame(rows)
    metrics = compute_metrics(results)

    tp, fp, tn, fn, accuracy, robustness = metrics
    print("\nFINAL METRICS")
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Robustness Score: {robustness:.3f}")

    return results, metrics
