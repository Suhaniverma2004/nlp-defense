import unicodedata
import re

# Emoji pattern
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F9FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001FA00-\U0001FAFF"
    "]+",
    flags=re.UNICODE
)

# Homoglyph mapping
HOMOGLYPH_MAP = {
    "а": "a",
    "е": "e",
    "о": "o",
    "р": "p",
    "с": "c",
    "у": "y",
    "х": "x",
}

def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(HOMOGLYPH_MAP.get(ch, ch) for ch in text)

def remove_emojis(text: str) -> str:
    return EMOJI_PATTERN.sub("", text)

def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())

def reconstruct_whitespace(text: str) -> str:
    tokens = text.split()

    # If text is split into characters → merge
    if len(tokens) > 0 and all(len(t) == 1 for t in tokens):
        return "".join(tokens)

    return text

def sanitize(text: str) -> str:
    text = normalize_unicode(text)
    text = remove_emojis(text)
    text = normalize_whitespace(text)
    text = reconstruct_whitespace(text)  # 🔥 Phase 3 improvement
    return text.strip()