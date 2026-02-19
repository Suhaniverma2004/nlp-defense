import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.attack_generator import generate_attacks

def main():
    with open("data/clean_prompts.txt") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
        print("Loaded prompts:", prompts)

    df = generate_attacks(prompts)
    df.to_csv("data/adversarial_prompts.csv", index=False)

    print("Adversarial dataset generated successfully.")
    print(df.head())

if __name__ == "__main__":
    main()

