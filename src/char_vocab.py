"""
Character Vocabulary Module

Build a character vocabulary for creating character-level embeddings.
Includes support for lowercase letters, uppercase letters, digits,
punctuation, and whitespace characters.
"""

import string


def build_char_vocab():
    """
    Build a character vocabulary dictionary.
    
    Returns:
        dict: char2idx mapping with special tokens
            - PAD token at index 0
            - Followed by all supported characters
    """
    # Special tokens
    vocab = {'<PAD>': 0}
    
    # Lowercase letters
    for char in string.ascii_lowercase:
        vocab[char] = len(vocab)
    
    # Uppercase letters
    for char in string.ascii_uppercase:
        vocab[char] = len(vocab)
    
    # Digits
    for char in string.digits:
        vocab[char] = len(vocab)
    
    # Punctuation
    for char in string.punctuation:
        vocab[char] = len(vocab)
    
    # Whitespace (space, tab, newline, etc.)
    whitespace_chars = [' ', '\t', '\n', '\r']
    for char in whitespace_chars:
        vocab[char] = len(vocab)
    
    return vocab


def get_char_idx(char, char_vocab):
    """
    Get the index of a character from vocabulary.
    
    Args:
        char (str): Single character
        char_vocab (dict): Character to index mapping
    
    Returns:
        int: Index of character, or 0 (PAD) if not in vocabulary
    """
    return char_vocab.get(char, 0)


def encode_text_to_char_ids(text, char_vocab):
    """
    Encode text to character IDs using the vocabulary.
    
    Args:
        text (str): Input text to encode
        char_vocab (dict): Character to index mapping
    
    Returns:
        list: List of character indices
    """
    return [get_char_idx(char, char_vocab) for char in text]


if __name__ == "__main__":
    # Example usage
    vocab = build_char_vocab()
    print(f"Character vocabulary size: {len(vocab)}")
    print(f"First 10 entries: {dict(list(vocab.items())[:10])}")
    
    # Test encoding
    test_text = "hello"
    char_ids = encode_text_to_char_ids(test_text, vocab)
    print(f"\nEncoding '{test_text}': {char_ids}")
