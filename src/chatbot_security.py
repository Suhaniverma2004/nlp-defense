from attack_generator import generate_all_attacks
from phase5_model import compute_similarity
from evaluate_defense import compute_metrics

THRESHOLD = 0.85

def classify(sim):
    return "PASS" if sim >= THRESHOLD else "FAIL"

def run_chatbot_security(user_input):

    attacks = generate_all_attacks(user_input)
    results = {}

    print("\n🔐 CHATBOT SECURITY SYSTEM\n")

    for attack_type, attacked_text in attacks.items():

        sim = compute_similarity(user_input, attacked_text)
        label = classify(sim)

        is_attack = attack_type != "clean"

        results[attack_type] = (sim, label, is_attack)

        print(f"[{attack_type.upper()}]")
        print(f"Text: {attacked_text}")
        print(f"Similarity: {sim:.3f} → {label}")
        print("-"*50)

    TP, FP, TN, FN, acc, rob = compute_metrics(results)

    print("\n📊 FINAL METRICS")
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Robustness Score: {rob:.3f}")