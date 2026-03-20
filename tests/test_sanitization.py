import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.sanitization import sanitize, normalize_unicode, remove_emojis, normalize_whitespace

def test_removes_intraword_emoji():
    result = sanitize("ha🔥ck acc💀ount")
    assert "🔥" not in result
    assert "💀" not in result
    assert "hack account" in result

def test_removes_between_word_emoji():
    result = sanitize("how 🔥 to 💀 hack")
    assert "🔥" not in result
    assert result == "how to hack"

def test_unicode_homoglyph_normalized():
    cyrillic_input = "h\u043Ew t\u043E h\u0430ck"
    result = normalize_unicode(cyrillic_input)
    assert result == "how to hack"

def test_whitespace_collapsed():
    result = normalize_whitespace("h a c k   a c c o u n t")
    assert "  " not in result

def test_combined_attack():
    combined = "hа🔥ck  а c c о u n t"
    result = sanitize(combined)
    assert "🔥" not in result
    assert "  " not in result

def test_clean_text_unchanged():
    clean = "how to hack account"
    assert sanitize(clean) == clean