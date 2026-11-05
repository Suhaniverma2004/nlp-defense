import pytest
from transformers import AutoTokenizer

from src.alignment import (
    AlignmentError,
    ExpandedSequenceOverflowError,
    TokenSequenceOverflowError,
    align_text,
    pad_alignment_units,
)


class FakeFastTokenizer:
    """Offline tokenizer fixture with explicitly supplied Python offsets."""

    is_fast = True

    def __init__(self, input_ids, offsets, special_tokens_mask):
        self.output = {
            "input_ids": input_ids,
            "offset_mapping": offsets,
            "special_tokens_mask": special_tokens_mask,
        }
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return self.output


def tokenizer_for(text, spans, *, ids=None, special_tokens_mask=None):
    """Make CLS/content/SEP outputs; spans are token IDs and offset pairs."""
    content_ids = ids if ids is not None else list(range(10, 10 + len(spans)))
    if special_tokens_mask is None:
        special_tokens_mask = [1] + [0] * len(spans) + [1]
    return FakeFastTokenizer(
        [101] + content_ids + [102],
        [(0, 0)] + list(spans) + [(0, 0)],
        special_tokens_mask,
    )


def assert_full_coverage(result, expected_text):
    assert result.text == expected_text
    assert "".join(span.text for span in result.source_spans) == expected_text
    cursor = 0
    for span in result.source_spans:
        assert span.start == cursor
        assert span.end >= span.start
        assert span.text == expected_text[span.start:span.end]
        cursor = span.end
    assert cursor == len(expected_text)
    assert result.coverage.partition_valid
    assert result.coverage.fully_represented
    raw_bytes = b"".join(
        bytes(byte_id - 1 for byte_id in span.byte_encoding.ids)
        for span in result.source_spans
    )
    assert raw_bytes == expected_text.encode("utf-8")


@pytest.mark.parametrize(
    "text",
    [
        "simple ASCII",
        "Latin кириллица 漢字",
        "pаypal",  # Cyrillic a and Latin letters
        "e\u0301",  # combining acute mark
        "left\u200bright",  # zero-width space
        "A\u200dB",  # zero-width joiner
        "💡",  # emoji
        "✈️",  # variation selector
        "👩‍👩‍👧‍👦",  # emoji ZWJ sequence
    ],
)
def test_full_raw_span_is_preserved_for_unicode_samples(text):
    tokenizer = tokenizer_for(text, [(0, len(text))])
    result = align_text(text, tokenizer, sample_id="unicode-case")

    assert_full_coverage(result, text)
    assert tokenizer.calls[0][0] == text
    assert tokenizer.calls[0][1]["truncation"] is False
    assert tokenizer.calls[0][1]["padding"] is False


def test_leading_inter_token_and_trailing_whitespace_are_separate_gaps():
    text = "  hello \t world\n"
    tokenizer = tokenizer_for(text, [(2, 7), (10, 15)])

    result = align_text(text, tokenizer)

    assert [(span.start, span.end, span.text) for span in result.gap_spans] == [
        (0, 2, "  "),
        (7, 10, " \t "),
        (15, 16, "\n"),
    ]
    assert [unit.kind for unit in result.units] == [
        "special", "gap", "token", "gap", "token", "gap", "special"
    ]
    assert_full_coverage(result, text)


def test_tokenizer_omitted_non_whitespace_source_is_an_explicit_gap():
    text = "aZb"
    tokenizer = tokenizer_for(text, [(0, 1), (2, 3)])

    result = align_text(text, tokenizer)

    assert [(gap.start, gap.end, gap.text) for gap in result.gap_spans] == [(1, 2, "Z")]
    assert_full_coverage(result, text)


def test_empty_input_has_only_structural_special_units_and_zero_source_bytes():
    tokenizer = tokenizer_for("", [])

    result = align_text("", tokenizer)

    assert result.token_spans == ()
    assert result.gap_spans == ()
    assert [unit.kind for unit in result.units] == ["special", "special"]
    assert all(unit.source_start == unit.source_end == 0 for unit in result.units)
    assert_full_coverage(result, "")


def test_nonempty_unknown_token_offset_encodes_original_not_literal_unk():
    text = "漢"
    tokenizer = tokenizer_for(
        text,
        [(0, 1)],
        ids=[100],
        special_tokens_mask=[1, 1, 1],
    )

    result = align_text(text, tokenizer)

    unknown = next(unit for unit in result.units if unit.kind == "token")
    assert unknown.byte_ids == tuple(byte + 1 for byte in text.encode("utf-8"))
    assert bytes(byte_id - 1 for byte_id in unknown.byte_ids).decode("utf-8") == text
    assert bytes(byte_id - 1 for byte_id in unknown.byte_ids) != b"[UNK]"
    assert_full_coverage(result, text)


def test_literal_cls_text_is_source_content_while_inserted_cls_sep_are_structural():
    text = "[CLS]"
    tokenizer = tokenizer_for(
        text,
        [(0, len(text))],
        ids=[101],
        special_tokens_mask=[1, 0, 1],
    )

    result = align_text(text, tokenizer)

    assert [(span.kind, span.start, span.end, span.text) for span in result.source_spans] == [
        ("token", 0, 5, "[CLS]"),
    ]
    literal = next(unit for unit in result.units if unit.kind == "token")
    assert literal.token_id == 101
    assert literal.byte_ids == tuple(byte + 1 for byte in b"[CLS]")
    assert sum(span.text.count("[") for span in result.source_spans) == 1
    assert sum(span.text.count("C") for span in result.source_spans) == 1
    assert sum(span.text.count("L") for span in result.source_spans) == 1
    assert sum(span.text.count("S") for span in result.source_spans) == 1
    assert sum(span.text.count("]") for span in result.source_spans) == 1
    inserted = [unit for unit in result.units if unit.kind == "special"]
    assert len(inserted) == 2
    assert all(unit.source_span_index is None for unit in inserted)
    assert all(unit.byte_ids == () and unit.byte_length == 0 for unit in inserted)
    assert_full_coverage(result, text)


@pytest.fixture(scope="module")
def cached_distilbert_fast_tokenizer():
    """Use only the locally cached intended tokenizer revision; never download."""
    revision = "12040accade4e8a0f71eabdb258fecc2e7e948be"
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased",
            revision=revision,
            use_fast=True,
            local_files_only=True,
        )
    except OSError as exc:
        pytest.skip(f"pinned tokenizer assets are not cached locally: {exc}")
    if not tokenizer.is_fast:
        pytest.fail("the cached distilbert-base-uncased tokenizer is not fast")
    return tokenizer


def test_cached_tokenizer_normalization_and_omitted_marks_keep_raw_bytes(cached_distilbert_fast_tokenizer):
    # BertNormalizer strips the combining accent from its token view. Its offset
    # ends at the base letter; the original combining mark must remain a gap.
    text = "e\u0301"
    result = align_text(text, cached_distilbert_fast_tokenizer)

    assert [(span.kind, span.start, span.end, span.text) for span in result.source_spans] == [
        ("token", 0, 1, "e"),
        ("gap", 1, 2, "\u0301"),
    ]
    assert_full_coverage(result, text)


def test_cached_tokenizer_unknown_token_uses_original_unicode_span(cached_distilbert_fast_tokenizer):
    text = "🧪"
    result = align_text(text, cached_distilbert_fast_tokenizer)

    assert result.input_ids[1] == cached_distilbert_fast_tokenizer.unk_token_id
    assert result.offsets[1] == (0, 1)
    assert result.special_tokens_mask[1] == 0
    unknown = next(unit for unit in result.units if unit.kind == "token")
    assert unknown.byte_ids == tuple(byte + 1 for byte in text.encode("utf-8"))
    assert_full_coverage(result, text)


def test_cached_tokenizer_literal_cls_spelling_has_source_offset(cached_distilbert_fast_tokenizer):
    text = "[CLS]"
    result = align_text(text, cached_distilbert_fast_tokenizer)

    assert result.offsets == ((0, 0), (0, 5), (0, 0))
    assert result.special_tokens_mask == (1, 0, 1)
    assert [unit.kind for unit in result.units] == ["special", "token", "special"]
    assert_full_coverage(result, text)


def test_cached_tokenizer_zero_width_only_input_is_retained_as_gap(cached_distilbert_fast_tokenizer):
    text = "\u200b"
    result = align_text(text, cached_distilbert_fast_tokenizer)

    assert result.gap_spans[0].text == text
    assert result.gap_spans[0].start == 0
    assert result.gap_spans[0].end == 1
    assert_full_coverage(result, text)


def test_structural_special_tokens_have_empty_source_spans_and_no_bytes():
    text = "word"
    tokenizer = tokenizer_for(text, [(0, 4)])

    result = align_text(text, tokenizer)
    specials = [unit for unit in result.units if unit.kind == "special"]

    assert len(specials) == 2
    assert all(unit.source_span_index is None for unit in specials)
    assert all(unit.byte_ids == () and unit.byte_length == 0 for unit in specials)
    assert all((start, end) == (0, 0) for start, end in result.unit_offsets if start == end)


def test_empty_content_token_span_is_retained_without_claiming_characters():
    text = "abc"
    tokenizer = tokenizer_for(text, [(0, 0), (0, 3)])

    result = align_text(text, tokenizer)

    assert len(result.token_spans) == 2
    assert (result.token_spans[0].start, result.token_spans[0].end) == (0, 0)
    assert result.token_spans[0].byte_encoding.ids == ()
    assert_full_coverage(result, text)


@pytest.mark.parametrize(
    "offsets",
    [
        [(-1, 1)],
        [(0, 4)],  # text length is three
        [(2, 1)],
    ],
)
def test_invalid_offset_bounds_or_order_are_rejected(offsets):
    tokenizer = tokenizer_for("abc", offsets)

    with pytest.raises(AlignmentError, match="invalid offset"):
        align_text("abc", tokenizer, sample_id="bad-offset")


def test_overlapping_token_offsets_are_rejected():
    tokenizer = tokenizer_for("abcd", [(0, 3), (2, 4)])

    with pytest.raises(AlignmentError, match="overlap or are out of order"):
        align_text("abcd", tokenizer, sample_id="overlap")


def test_slow_tokenizers_are_rejected_before_tokenization():
    class SlowTokenizer:
        is_fast = False

    with pytest.raises(AlignmentError, match="fast tokenizer"):
        align_text("text", SlowTokenizer())


def test_inconsistent_tokenizer_field_lengths_are_rejected():
    tokenizer = FakeFastTokenizer([101, 10, 102], [(0, 0), (0, 3)], [1, 0, 1])

    with pytest.raises(AlignmentError, match="lengths differ"):
        align_text("abc", tokenizer)


def test_token_truncation_is_strict_by_default_and_reports_omitted_range_when_allowed():
    text = "aa bb cc"
    tokenizer = tokenizer_for(text, [(0, 2), (3, 5), (6, 8)])

    with pytest.raises(TokenSequenceOverflowError, match="max_token_length=3"):
        align_text(text, tokenizer, max_token_length=3)

    result = align_text(
        text,
        tokenizer,
        max_token_length=3,
        allow_truncation=True,
    )
    assert result.coverage.token_truncated
    assert [
        (item.start, item.end, item.reason)
        for item in result.coverage.unrepresented_ranges
    ] == [(2, 8, "token_truncation")]
    assert result.units[-1].kind == "special"
    assert not result.coverage.fully_represented


def test_expanded_sequence_overflow_is_strict_or_explicitly_reported():
    text = "a b"
    tokenizer = tokenizer_for(text, [(0, 1), (2, 3)])

    with pytest.raises(ExpandedSequenceOverflowError, match="max_expanded_length=4"):
        align_text(text, tokenizer, max_expanded_length=4)

    result = align_text(
        text,
        tokenizer,
        max_expanded_length=4,
        allow_truncation=True,
    )
    assert result.coverage.expanded_truncated
    assert [(item.start, item.end, item.reason) for item in result.coverage.unrepresented_ranges] == [
        (2, 3, "expanded_sequence_truncation")
    ]


def test_byte_span_limit_is_strict_and_explicit_cropping_is_reported():
    text = "éx"
    tokenizer = tokenizer_for(text, [(0, 2)])

    with pytest.raises(ValueError, match="max_bytes=1"):
        align_text(text, tokenizer, max_bytes_per_span=1)

    result = align_text(
        text,
        tokenizer,
        max_bytes_per_span=1,
        allow_byte_truncation=True,
    )
    assert result.coverage.byte_truncated
    assert [(item.start, item.end, item.reason) for item in result.coverage.unrepresented_ranges] == [
        (0, 2, "byte_span_truncation")
    ]


def test_expanded_batch_padding_uses_empty_spans_and_zero_mask():
    short_text = "a"
    long_text = "a b"
    short = align_text(short_text, tokenizer_for(short_text, [(0, 1)]))
    long = align_text(long_text, tokenizer_for(long_text, [(0, 1), (2, 3)]))

    padded = pad_alignment_units([short, long])
    padding_units = [unit for unit in padded.units[0] if unit.kind == "padding"]

    assert len(padded.units[0]) == len(padded.units[1])
    assert len(padding_units) == 2
    assert padding_units[0].source_span_index is None
    assert (padding_units[0].source_start, padding_units[0].source_end) == (0, 0)
    assert padding_units[0].byte_ids == ()
    assert padded.attention_mask[0] == (1, 1, 1, 0, 0)
    assert padded.attention_mask[1] == (1, 1, 1, 1, 1)
