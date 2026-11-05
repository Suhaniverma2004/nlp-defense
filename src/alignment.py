"""Fast-tokenizer source alignment with explicit raw-text gap units.

Offsets are interpreted as half-open Python-string character ranges into the
exact input string passed to the tokenizer. This module does not normalize text
or reconstruct source spans from WordPiece token strings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from src.byte_encoding import ByteEncoding, encode_utf8_bytes
from src.unicode_policy import SampleId, validate_text

UnitKind = Literal["token", "gap", "special", "padding"]


class AlignmentError(ValueError):
    """Raised when tokenizer output cannot be safely aligned to raw text."""


class TokenSequenceOverflowError(AlignmentError):
    """Raised when the tokenizer sequence exceeds an explicit strict limit."""


class ExpandedSequenceOverflowError(AlignmentError):
    """Raised when token plus gap units exceed the model-position limit."""


@dataclass(frozen=True)
class SourceSpan:
    """A token-owned or separate gap-owned half-open source interval."""

    span_index: int
    kind: Literal["token", "gap"]
    start: int
    end: int
    text: str
    token_index: int | None
    boundary_token_index: int | None
    byte_encoding: ByteEncoding


@dataclass(frozen=True)
class AlignmentUnit:
    """An ordered model unit; specials have no source span or byte content."""

    kind: UnitKind
    token_id: int | None
    token_index: int | None
    source_span_index: int | None
    source_start: int
    source_end: int
    byte_ids: tuple[int, ...]
    byte_length: int


@dataclass(frozen=True)
class UnrepresentedRange:
    """A source range explicitly omitted from final model units and its cause."""

    start: int
    end: int
    reason: str


@dataclass(frozen=True)
class CoverageReport:
    """Source coverage and truncation metadata for one aligned input."""

    sample_id: SampleId
    source_length_chars: int
    source_length_bytes: int
    token_span_count: int
    gap_span_count: int
    expanded_sequence_length: int
    token_truncated: bool
    expanded_truncated: bool
    byte_truncated: bool
    partition_valid: bool
    unrepresented_ranges: tuple[UnrepresentedRange, ...]

    @property
    def fully_represented(self) -> bool:
        """Whether every source character is present in final byte units."""
        return self.partition_valid and not self.unrepresented_ranges


@dataclass(frozen=True)
class AlignmentResult:
    """Tokenization, source spans, ordered units, and byte coverage for text.

    ``input_ids`` and ``offsets`` preserve the complete untruncated tokenizer
    result for audit. ``units`` is the actual sequence after any explicitly
    enabled token/expanded-length truncation and is the model-facing ordering.
    """

    text: str
    input_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    special_tokens_mask: tuple[int, ...]
    token_spans: tuple[SourceSpan, ...]
    gap_spans: tuple[SourceSpan, ...]
    source_spans: tuple[SourceSpan, ...]
    units: tuple[AlignmentUnit, ...]
    coverage: CoverageReport

    @property
    def attention_mask(self) -> tuple[int, ...]:
        """Return an unpadded all-one mask for the expanded ordered units."""
        return (1,) * len(self.units)

    @property
    def model_unit_token_ids(self) -> tuple[int | None, ...]:
        """Return token IDs aligned to model units; gap units have no token ID."""
        return tuple(unit.token_id for unit in self.units)

    @property
    def unit_offsets(self) -> tuple[tuple[int, int], ...]:
        """Return source offsets; structural specials use the empty range (0, 0)."""
        return tuple((unit.source_start, unit.source_end) for unit in self.units)


@dataclass(frozen=True)
class PaddedAlignmentBatch:
    """Right-padded model units and masks for a batch of aligned examples."""

    units: tuple[tuple[AlignmentUnit, ...], ...]
    attention_mask: tuple[tuple[int, ...], ...]


def _as_single_list(value: Any, name: str, *, pairs: bool = False) -> list[Any]:
    """Convert tokenizer outputs to one example's flat Python list."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise AlignmentError(f"Tokenizer field {name!r} must be a list-like value.")

    values = list(value)
    if pairs:
        is_batched = bool(values) and isinstance(values[0], (list, tuple)) and bool(values[0]) and isinstance(values[0][0], (list, tuple))
    else:
        is_batched = bool(values) and isinstance(values[0], (list, tuple))
    if is_batched:
        if len(values) != 1:
            raise AlignmentError(
                f"Tokenizer field {name!r} returned {len(values)} examples; "
                "align_text expects exactly one string."
            )
        values = list(values[0])
    return values


def _make_span(
    *,
    span_index: int,
    kind: Literal["token", "gap"],
    text: str,
    start: int,
    end: int,
    sample_id: SampleId,
    max_bytes_per_span: int | None,
    allow_byte_truncation: bool,
    token_index: int | None = None,
    boundary_token_index: int | None = None,
) -> SourceSpan:
    source_text = text[start:end]
    encoding = encode_utf8_bytes(
        source_text,
        max_bytes=max_bytes_per_span,
        allow_truncation=allow_byte_truncation,
        source_start=start,
        sample_id=sample_id,
    )
    return SourceSpan(
        span_index=span_index,
        kind=kind,
        start=start,
        end=end,
        text=source_text,
        token_index=token_index,
        boundary_token_index=boundary_token_index,
        byte_encoding=encoding,
    )


def _check_partition(text: str, spans: Sequence[SourceSpan], sample_id: SampleId) -> None:
    cursor = 0
    pieces = []
    for span in spans:
        if span.start != cursor:
            raise AlignmentError(
                f"sample {sample_id!r}: internal source partition error at "
                f"character index {cursor}; next span is [{span.start}, {span.end})."
            )
        if span.end < span.start:
            raise AlignmentError(
                f"sample {sample_id!r}: reversed source span [{span.start}, {span.end})."
            )
        pieces.append(span.text)
        cursor = span.end

    if cursor != len(text) or "".join(pieces) != text:
        raise AlignmentError(
            f"sample {sample_id!r}: token and gap spans do not reconstruct the "
            f"original string exactly (covered {cursor} of {len(text)} characters)."
        )


def _merge_unrepresented(
    ranges: Sequence[UnrepresentedRange],
) -> tuple[UnrepresentedRange, ...]:
    ordered = sorted((item for item in ranges if item.start < item.end), key=lambda item: (item.start, item.end))
    merged: list[UnrepresentedRange] = []
    for item in ordered:
        if merged and item.start < merged[-1].end:
            raise AlignmentError(
                f"Unrepresented source ranges overlap: [{merged[-1].start}, {merged[-1].end}) "
                f"and [{item.start}, {item.end})."
            )
        if merged and item.start == merged[-1].end and item.reason == merged[-1].reason:
            prior = merged[-1]
            merged[-1] = UnrepresentedRange(prior.start, item.end, prior.reason)
        else:
            merged.append(item)
    return tuple(merged)


def align_text(
    text: object,
    tokenizer: Any,
    *,
    sample_id: SampleId = None,
    max_token_length: int | None = None,
    max_expanded_length: int | None = None,
    max_bytes_per_span: int | None = None,
    allow_truncation: bool = False,
    allow_byte_truncation: bool = False,
) -> AlignmentResult:
    """Align exact source spans to fast-tokenizer tokens and separate gap units.

    The tokenizer receives the unchanged original string with token truncation
    and padding disabled. Gaps are the complement of non-empty content-token
    offsets and are never merged into neighboring token spans. Strict overflow
    behavior is the default; explicit truncation is reported by source range.

    ``max_token_length`` counts tokenizer output positions including structural
    specials. ``max_expanded_length`` counts model units after adding gaps.
    Byte limits apply independently to every token and gap span.
    """
    source = validate_text(text, sample_id=sample_id)
    label = "sample <unspecified>" if sample_id is None else f"sample {sample_id!r}"

    if getattr(tokenizer, "is_fast", False) is not True:
        raise AlignmentError(
            f"{label}: a fast tokenizer with offset_mapping support is required; "
            "slow tokenizers cannot provide source alignment."
        )
    for name, limit in (
        ("max_token_length", max_token_length),
        ("max_expanded_length", max_expanded_length),
        ("max_bytes_per_span", max_bytes_per_span),
    ):
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer or None.")

    try:
        encoded = tokenizer(
            source,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            truncation=False,
            padding=False,
        )
    except Exception as exc:
        raise AlignmentError(
            f"{label}: fast tokenizer failed to return offsets for the unchanged input: {exc}"
        ) from exc
    if not isinstance(encoded, Mapping):
        raise AlignmentError(f"{label}: tokenizer output must be a mapping.")

    missing = {"input_ids", "offset_mapping", "special_tokens_mask"} - set(encoded)
    if missing:
        raise AlignmentError(f"{label}: tokenizer output is missing fields {sorted(missing)}.")

    ids = _as_single_list(encoded["input_ids"], "input_ids")
    raw_offsets = _as_single_list(encoded["offset_mapping"], "offset_mapping", pairs=True)
    special_mask = _as_single_list(encoded["special_tokens_mask"], "special_tokens_mask")
    if not (len(ids) == len(raw_offsets) == len(special_mask)):
        raise AlignmentError(
            f"{label}: input_ids ({len(ids)}), offsets ({len(raw_offsets)}), and "
            f"special_tokens_mask ({len(special_mask)}) lengths differ."
        )

    normalized_ids: list[int] = []
    offsets: list[tuple[int, int]] = []
    normalized_special_mask: list[int] = []
    for token_index, (token_id, pair, special_flag) in enumerate(zip(ids, raw_offsets, special_mask)):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise AlignmentError(f"{label}: token ID at index {token_index} is not an integer.")
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise AlignmentError(
                f"{label}: offset at token {token_index} must be a (start, end) pair."
            )
        start, end = pair
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise AlignmentError(f"{label}: offsets at token {token_index} must be integers.")
        if start < 0 or end < start or end > len(source):
            raise AlignmentError(
                f"{label}: invalid offset [{start}, {end}) at token {token_index}; "
                f"expected 0 <= start <= end <= {len(source)} for the original Python string."
            )
        if isinstance(special_flag, bool):
            special_flag = int(special_flag)
        if not isinstance(special_flag, int) or special_flag not in (0, 1):
            raise AlignmentError(
                f"{label}: special-token flag at index {token_index} must be 0 or 1."
            )
        normalized_ids.append(token_id)
        offsets.append((start, end))
        normalized_special_mask.append(special_flag)

    # Empty flagged positions are structural specials. A non-empty [UNK] offset
    # remains a content span even if the tokenizer includes its ID in specials.
    structural = [
        bool(flag and start == end)
        for flag, (start, end) in zip(normalized_special_mask, offsets)
    ]
    content_indices = [index for index, is_special in enumerate(structural) if not is_special]
    structural_indices = [index for index, is_special in enumerate(structural) if is_special]

    previous_end = 0
    token_span_by_index: dict[int, SourceSpan] = {}
    token_spans: list[SourceSpan] = []
    gap_spans: list[SourceSpan] = []
    source_spans: list[SourceSpan] = []
    gap_before_token: dict[int, SourceSpan] = {}
    cursor = 0

    for token_index in content_indices:
        start, end = offsets[token_index]
        if start < previous_end:
            raise AlignmentError(
                f"{label}: token offsets overlap or are out of order at token {token_index}: "
                f"[{start}, {end}) begins before previous content end {previous_end}."
            )
        if start < cursor:
            raise AlignmentError(
                f"{label}: token offset [{start}, {end}) at index {token_index} overlaps "
                f"already-owned source through character {cursor}."
            )
        if start > cursor:
            gap = _make_span(
                span_index=len(source_spans),
                kind="gap",
                text=source,
                start=cursor,
                end=start,
                sample_id=sample_id,
                max_bytes_per_span=max_bytes_per_span,
                allow_byte_truncation=allow_byte_truncation,
                boundary_token_index=token_index,
            )
            gap_spans.append(gap)
            source_spans.append(gap)
            gap_before_token[token_index] = gap

        token_span = _make_span(
            span_index=len(source_spans),
            kind="token",
            text=source,
            start=start,
            end=end,
            sample_id=sample_id,
            max_bytes_per_span=max_bytes_per_span,
            allow_byte_truncation=allow_byte_truncation,
            token_index=token_index,
        )
        token_spans.append(token_span)
        token_span_by_index[token_index] = token_span
        source_spans.append(token_span)
        cursor = end
        if start < end:
            previous_end = end
        else:
            previous_end = start

    trailing_gap: SourceSpan | None = None
    if cursor < len(source):
        trailing_gap = _make_span(
            span_index=len(source_spans),
            kind="gap",
            text=source,
            start=cursor,
            end=len(source),
            sample_id=sample_id,
            max_bytes_per_span=max_bytes_per_span,
            allow_byte_truncation=allow_byte_truncation,
            boundary_token_index=None,
        )
        gap_spans.append(trailing_gap)
        source_spans.append(trailing_gap)

    # If the tokenizer emitted no content tokens, the entire non-empty source
    # becomes one explicit gap unit, positioned between prefix/suffix specials.
    _check_partition(source, source_spans, sample_id)

    if content_indices:
        first_content = content_indices[0]
        last_content = content_indices[-1]
        internal_specials = [
            index for index in structural_indices if first_content < index < last_content
        ]
        if internal_specials:
            raise AlignmentError(
                f"{label}: structural special tokens {internal_specials} occur between "
                "content tokens; their placement relative to source spans is ambiguous."
            )
        prefix_specials = [index for index in structural_indices if index < first_content]
        suffix_specials = [index for index in structural_indices if index > last_content]
    elif len(structural_indices) >= 2:
        prefix_specials, suffix_specials = structural_indices[:-1], structural_indices[-1:]
    elif structural_indices:
        prefix_specials, suffix_specials = [], structural_indices
    else:
        prefix_specials, suffix_specials = [], []

    all_units: list[AlignmentUnit] = []
    for special_index in prefix_specials:
        all_units.append(AlignmentUnit(
            kind="special",
            token_id=normalized_ids[special_index],
            token_index=special_index,
            source_span_index=None,
            source_start=0,
            source_end=0,
            byte_ids=(),
            byte_length=0,
        ))

    def add_gap(gap: SourceSpan) -> None:
        all_units.append(AlignmentUnit(
            kind="gap",
            token_id=None,
            token_index=None,
            source_span_index=gap.span_index,
            source_start=gap.start,
            source_end=gap.end,
            byte_ids=gap.byte_encoding.ids,
            byte_length=gap.byte_encoding.length,
        ))

    for token_index in content_indices:
        gap = gap_before_token.get(token_index)
        if gap is not None:
            add_gap(gap)
        span = token_span_by_index[token_index]
        all_units.append(AlignmentUnit(
            kind="token",
            token_id=normalized_ids[token_index],
            token_index=token_index,
            source_span_index=span.span_index,
            source_start=span.start,
            source_end=span.end,
            byte_ids=span.byte_encoding.ids,
            byte_length=span.byte_encoding.length,
        ))

    if not content_indices:
        for gap in gap_spans:
            add_gap(gap)
    elif trailing_gap is not None:
        add_gap(trailing_gap)

    for special_index in suffix_specials:
        all_units.append(AlignmentUnit(
            kind="special",
            token_id=normalized_ids[special_index],
            token_index=special_index,
            source_span_index=None,
            source_start=0,
            source_end=0,
            byte_ids=(),
            byte_length=0,
        ))

    # A special token not in the leading/trailing groups would otherwise be
    # silently lost; reject rather than guess where its source-independent unit
    # belongs.
    included_specials = set(prefix_specials) | set(suffix_specials)
    if included_specials != set(structural_indices):
        raise AlignmentError(f"{label}: unsupported structural special-token ordering.")

    unrepresented_reasons: dict[int, str] = {}
    token_truncated = False
    if max_token_length is not None and len(normalized_ids) > max_token_length:
        if not allow_truncation:
            raise TokenSequenceOverflowError(
                f"{label}: tokenizer produced {len(normalized_ids)} positions, exceeding "
                f"max_token_length={max_token_length}; increase the limit or explicitly "
                "enable truncation."
            )
        token_truncated = True
        content_budget = max_token_length - len(structural_indices)
        if content_budget < 0:
            raise TokenSequenceOverflowError(
                f"{label}: max_token_length={max_token_length} cannot retain "
                f"{len(structural_indices)} structural special tokens."
            )
        retained_content = set(content_indices[:content_budget])
        dropped_content = set(content_indices[content_budget:])
        filtered_units = []
        for unit in all_units:
            if unit.kind == "token" and unit.token_index in dropped_content:
                unrepresented_reasons[unit.source_span_index] = "token_truncation"
                continue
            if unit.kind == "gap":
                span = source_spans[unit.source_span_index]
                boundary = span.boundary_token_index
                if boundary is not None and boundary not in retained_content:
                    unrepresented_reasons[span.span_index] = "token_truncation"
                    continue
                if boundary is None and content_indices:
                    unrepresented_reasons[span.span_index] = "token_truncation"
                    continue
            filtered_units.append(unit)
        all_units = filtered_units

    expanded_truncated = False
    if max_expanded_length is not None and len(all_units) > max_expanded_length:
        if not allow_truncation:
            raise ExpandedSequenceOverflowError(
                f"{label}: {len(all_units)} token/gap/special units exceed "
                f"max_expanded_length={max_expanded_length}; increase the limit or "
                "explicitly enable truncation."
            )
        expanded_truncated = True
        prefix_count = 0
        while prefix_count < len(all_units) and all_units[prefix_count].kind == "special":
            prefix_count += 1
        suffix_count = 0
        while suffix_count < len(all_units) - prefix_count and all_units[-1 - suffix_count].kind == "special":
            suffix_count += 1
        if prefix_count + suffix_count > max_expanded_length:
            raise ExpandedSequenceOverflowError(
                f"{label}: max_expanded_length={max_expanded_length} cannot retain "
                "the tokenizer's structural special tokens."
            )
        middle_budget = max_expanded_length - prefix_count - suffix_count
        middle_end = len(all_units) - suffix_count if suffix_count else len(all_units)
        retained = all_units[:prefix_count] + all_units[prefix_count:prefix_count + middle_budget]
        retained += all_units[middle_end:]
        for unit in all_units[prefix_count + middle_budget:middle_end]:
            if unit.source_span_index is not None:
                unrepresented_reasons[unit.source_span_index] = "expanded_sequence_truncation"
        all_units = retained

    retained_span_indices = {
        unit.source_span_index for unit in all_units if unit.source_span_index is not None
    }
    unrepresented: list[UnrepresentedRange] = []
    byte_truncated = any(span.byte_encoding.truncated for span in source_spans)
    for span in source_spans:
        if span.start == span.end:
            continue
        if span.span_index not in retained_span_indices:
            reason = unrepresented_reasons.get(span.span_index, "not_in_model_units")
            unrepresented.append(UnrepresentedRange(span.start, span.end, reason))
        elif span.byte_encoding.omitted_char_range is not None:
            start, end = span.byte_encoding.omitted_char_range
            unrepresented.append(UnrepresentedRange(start, end, "byte_span_truncation"))

    unrepresented_ranges = _merge_unrepresented(unrepresented)
    unapproved_ranges = [
        item
        for item in unrepresented_ranges
        if not (
            (item.reason == "byte_span_truncation" and allow_byte_truncation)
            or (item.reason != "byte_span_truncation" and allow_truncation)
        )
    ]
    if unapproved_ranges:
        raise AlignmentError(
            f"{label}: strict source coverage failed; unrepresented ranges: "
            f"{[(item.start, item.end, item.reason) for item in unapproved_ranges]}."
        )

    coverage = CoverageReport(
        sample_id=sample_id,
        source_length_chars=len(source),
        source_length_bytes=len(source.encode("utf-8", errors="strict")),
        token_span_count=len(token_spans),
        gap_span_count=len(gap_spans),
        expanded_sequence_length=len(all_units),
        token_truncated=token_truncated,
        expanded_truncated=expanded_truncated,
        byte_truncated=byte_truncated,
        partition_valid=True,
        unrepresented_ranges=unrepresented_ranges,
    )

    if not any(span.byte_encoding.truncated for span in source_spans):
        encoded_source = b"".join(
            bytes(value - 1 for value in span.byte_encoding.ids)
            for span in source_spans
        )
        if encoded_source != source.encode("utf-8", errors="strict"):
            raise AlignmentError(f"{label}: UTF-8 byte coverage did not reconstruct the input.")

    return AlignmentResult(
        text=source,
        input_ids=tuple(normalized_ids),
        offsets=tuple(offsets),
        special_tokens_mask=tuple(normalized_special_mask),
        token_spans=tuple(token_spans),
        gap_spans=tuple(gap_spans),
        source_spans=tuple(source_spans),
        units=tuple(all_units),
        coverage=coverage,
    )


def pad_alignment_units(
    results: Sequence[AlignmentResult],
    *,
    pad_to_length: int | None = None,
    pad_token_id: int = 0,
) -> PaddedAlignmentBatch:
    """Right-pad expanded alignments with empty-source padding units.

    Padding never truncates. Every synthetic padding unit has no source span,
    no byte IDs, an empty ``(0, 0)`` offset, and attention-mask value zero.
    """
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
        raise ValueError("pad_token_id must be an integer.")
    if any(not isinstance(result, AlignmentResult) for result in results):
        raise TypeError("results must contain AlignmentResult instances.")

    longest = max((len(result.units) for result in results), default=0)
    if pad_to_length is None:
        width = longest
    else:
        if (
            isinstance(pad_to_length, bool)
            or not isinstance(pad_to_length, int)
            or pad_to_length < 0
        ):
            raise ValueError("pad_to_length must be a non-negative integer or None.")
        if pad_to_length < longest:
            raise ValueError(
                f"pad_to_length={pad_to_length} is shorter than the longest expanded "
                f"alignment ({longest}); padding never truncates."
            )
        width = pad_to_length

    padded_rows = []
    masks = []
    for result in results:
        real_units = result.units
        pad_count = width - len(real_units)
        padding_units = tuple(
            AlignmentUnit(
                kind="padding",
                token_id=pad_token_id,
                token_index=None,
                source_span_index=None,
                source_start=0,
                source_end=0,
                byte_ids=(),
                byte_length=0,
            )
            for _ in range(pad_count)
        )
        padded_rows.append(real_units + padding_units)
        masks.append((1,) * len(real_units) + (0,) * pad_count)

    return PaddedAlignmentBatch(tuple(padded_rows), tuple(masks))
