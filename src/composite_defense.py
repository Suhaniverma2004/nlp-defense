def fix_obfuscation(text):
    return (text.replace("@", "a")
                .replace("0", "o")
                .replace("3", "e"))

def composite_clean(text):

    for _ in range(2):

        # fix obfuscation
        text = (text.replace("@", "a")
                    .replace("0", "o")
                    .replace("3", "e"))

        # remove emojis
        text = "".join(c for c in text if c.isalnum() or c.isspace())

        # fix spacing
        tokens = text.split()
        if all(len(t) == 1 for t in tokens):
            text = "".join(tokens)
        else:
            text = " ".join(tokens)

    return text

def is_composite(text):
    score = 0

    if any(c in text for c in "@03"): score += 1
    if any(e in text for e in ["🔥","💀","💣","⚠️","🎯","😊"]): score += 1
    if " " in text and len(text.split()) > len("".join(text.split())): score += 1

    return score >= 2