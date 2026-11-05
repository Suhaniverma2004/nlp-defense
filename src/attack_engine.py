def emoji_insertion(text):
    return text.replace("a", "a😊")

def emoji_replacement(text):
    return text.replace("love", "❤️")

def unicode_attack(text):
    return text.replace("a", "а")  # Cyrillic

def whitespace_attack(text):
    return " ".join(list(text))

def repeated_chars(text):
    return text.replace("o", "oooo")

def punctuation_noise(text):
    return text + "!!!???"

def case_flip(text):
    return text.swapcase()

def mixed_attack(text):
    text = emoji_insertion(text)
    text = unicode_attack(text)
    return whitespace_attack(text)


TRANSFORMATIONS = {
    "Emoji Insertion": emoji_insertion,
    "Emoji Replacement": emoji_replacement,
    "Unicode Homoglyph Substitution": unicode_attack,
    "Whitespace Spacing": whitespace_attack,
    "Character Repetition": repeated_chars,
    "Punctuation Noise": punctuation_noise,
    "Case Change": case_flip,
    "Mixed Transformation": mixed_attack,
}

# Historical mapping retained for callers that still use the old labels.
ATTACKS = {
    "Emoji Insertion": emoji_insertion,
    "Emoji Replacement": emoji_replacement,
    "Unicode Attack": unicode_attack,
    "Whitespace Attack": whitespace_attack,
    "Repeated Characters": repeated_chars,
    "Punctuation Noise": punctuation_noise,
    "Case Flip": case_flip,
    "Mixed Attack (Hard)": mixed_attack,
}
