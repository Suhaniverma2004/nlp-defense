import random
import pandas as pd

# Unicode homoglyph map (simplified)
HOMOGLYPHS = {
    'a': 'а',  # Cyrillic a
    'o': 'о',  # Cyrillic o
    'e': 'е',
    'c': 'с'
}

EMOJIS = ["🔥", "💀", "🎯", "⚠️", "💣"]

def unicode_attack(text):
    attacked = ""
    for ch in text:
        if ch in HOMOGLYPHS and random.random() < 0.5:
            attacked += HOMOGLYPHS[ch]
        else:
            attacked += ch
    return attacked

def emoji_attack(text):
    words = text.split()
    attacked = []
    for w in words:
        attacked.append(w)
        if random.random() < 0.4:
            attacked.append(random.choice(EMOJIS))
    return " ".join(attacked)

def whitespace_attack(text):
    return " ".join(list(text))

def generate_attacks(clean_prompts):
    data = []
    for prompt in clean_prompts:
        data.append(("unicode", prompt, unicode_attack(prompt)))
        data.append(("emoji", prompt, emoji_attack(prompt)))
        data.append(("whitespace", prompt, whitespace_attack(prompt)))
    return pd.DataFrame(data, columns=["type", "clean", "adversarial"])
