def fix_obfuscation(text):
    return (text.replace("@", "a")
                .replace("0", "o")
                .replace("3", "e"))

def composite_clean(text):
    for _ in range(2):
        text = fix_obfuscation(text)
        text = " ".join(text.split())
    return text

def is_composite(text):
    score = 0
    if any(c in text for c in "@03"): score += 1
    if len(text.split()) > len("".join(text.split())): score += 1
    return score >= 2