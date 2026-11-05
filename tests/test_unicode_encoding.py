import pytest

from src.byte_encoding import (
    BYTE_VOCAB_SIZE,
    PAD_ID,
    ByteSpanOverflowError,
    encode_utf8_bytes,
    pad_byte_sequences,
)
from src.unicode_policy import UnicodeInputError, validate_text, validate_texts


@pytest.mark.parametrize(
    "text",
    [
        "ASCII text 123!",
        "Latin Ελληνικά кириллица 日本語",
        "pаypal",  # Cyrillic small a between Latin letters
        "e\u0301",  # decomposed combining acute
        "left\u200bright",  # zero-width space
        "A\u200dB",  # zero-width joiner
        "Hello 👩‍💻",  # emoji with a ZWJ sequence
        "✈️",  # variation selector
        "👩‍👩‍👧‍👦",  # family emoji ZWJ sequence
    ],
)
def test_utf8_byte_ids_reconstruct_exact_original_text(text):
    result = encode_utf8_bytes(text)

    assert result.ids == tuple(byte + 1 for byte in text.encode("utf-8"))
    assert result.length == len(text.encode("utf-8"))
    assert all(1 <= byte_id < BYTE_VOCAB_SIZE for byte_id in result.ids)
    reconstructed = bytes(byte_id - 1 for byte_id in result.ids).decode("utf-8")
    assert reconstructed == text


def test_validation_returns_the_exact_string_without_normalizing():
    decomposed = "e\u0301"
    checked = validate_text(decomposed, sample_id="unicode-1")

    assert checked is decomposed
    assert checked == decomposed
    assert checked != "é"
    assert validate_texts([" a ", "💡"], sample_ids=[1, "two"]) == (" a ", "💡")


@pytest.mark.parametrize("value", [None, 1, b"text", ["text"]])
def test_non_string_inputs_are_rejected_with_sample_context(value):
    with pytest.raises(TypeError, match="sample 'row-4'.*must be str"):
        validate_text(value, sample_id="row-4")


@pytest.mark.parametrize("surrogate", ["\ud800", "x\udfffy"])
def test_unpaired_surrogates_are_rejected_with_character_position(surrogate):
    with pytest.raises(UnicodeInputError, match=r"sample 'row-7'.*character index"):
        validate_text(surrogate, sample_id="row-7")


def test_byte_ids_are_shifted_and_padding_is_unambiguous():
    padded = pad_byte_sequences(
        [encode_utf8_bytes("A"), encode_utf8_bytes("é")],
        pad_to_length=3,
    )

    assert padded.input_ids == ((66, PAD_ID, PAD_ID), (196, 170, PAD_ID))
    assert padded.lengths == (1, 2)
    assert padded.attention_mask == ((1, 0, 0), (1, 1, 0))
    assert PAD_ID == 0


def test_padding_never_truncates_and_rejects_pad_inside_unpadded_data():
    with pytest.raises(ValueError, match="padding never truncates"):
        pad_byte_sequences([[2, 3]], pad_to_length=1)

    with pytest.raises(ValueError, match="PAD ID inside unpadded data"):
        pad_byte_sequences([[65, 0, 66]])


def test_byte_overflow_is_strict_by_default():
    with pytest.raises(ByteSpanOverflowError, match="sample 'long-token'.*max_bytes=2"):
        encode_utf8_bytes("éa", max_bytes=2, sample_id="long-token")


def test_explicit_byte_truncation_keeps_codepoints_whole_and_reports_range():
    result = encode_utf8_bytes(
        "A😀B",
        max_bytes=4,
        allow_truncation=True,
        source_start=10,
        sample_id="cropped",
    )

    assert result.ids == tuple(byte + 1 for byte in b"A")
    assert result.represented_char_end == 11
    assert result.omitted_char_range == (11, 13)
    assert result.truncated


def test_empty_batch_padding_has_deterministic_empty_shape():
    padded = pad_byte_sequences([])

    assert padded.input_ids == ()
    assert padded.lengths == ()
    assert padded.attention_mask == ()
