"""
Character Vocabulary Module

Build a character vocabulary for character-level embeddings.
Supports lowercase, uppercase, digits, punctuation, and whitespace.
Includes special tokens for padding and unknown characters.
"""

import string


def build_char_vocab():
    """
    Build character vocabulary.

    Returns:
        dict: char → index mapping
            <PAD> = 0
            <UNK> = 1
    """

    # Special tokens
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1
    }

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

    # Whitespace characters
    whitespace_chars = [' ', '\t', '\n', '\r']
    for char in whitespace_chars:
        vocab[char] = len(vocab)

    return vocab


def get_char_idx(char, char_vocab):
    """
    Get index of a character.

    Args:
        char (str): single character
        char_vocab (dict): vocabulary

    Returns:
        int: index of character
    """

    # Normalize to lowercase to reduce sparsity
    char = char.lower()

    # Return UNK if character not found
    return char_vocab.get(char, char_vocab['<UNK>'])


def encode_text_to_char_ids(text, char_vocab):
    """
    Encode text into character IDs.

    Args:
        text (str): input text
        char_vocab (dict): vocabulary

    Returns:
        list[int]: list of character indices
    """
    return [get_char_idx(c, char_vocab) for c in text]


if __name__ == "__main__":
    vocab = build_char_vocab()

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Sample entries: {list(vocab.items())[:10]}")

    test_text = "Hello@123"
    char_ids = encode_text_to_char_ids(test_text, vocab)

    print(f"\nText: {test_text}")
    print(f"Encoded: {char_ids}")