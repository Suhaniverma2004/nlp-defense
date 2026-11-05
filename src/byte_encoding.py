"""Lossless UTF-8 byte IDs and deterministic byte-sequence padding."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from src.unicode_policy import SampleId, validate_text

PAD_ID = 0
BYTE_ID_OFFSET = 1
BYTE_VOCAB_SIZE = 257


class ByteSpanOverflowError(ValueError):
    """Raised when a byte span exceeds its configured limit in strict mode."""


@dataclass(frozen=True)
class ByteEncoding:
    """Encoded byte IDs and any source characters omitted by explicit cropping."""

    ids: tuple[int, ...]
    length: int
    source_start: int
    source_end: int
    represented_char_end: int
    omitted_char_range: tuple[int, int] | None

    @property
    def truncated(self) -> bool:
        """Whether explicit byte cropping omitted any source characters."""
        return self.omitted_char_range is not None


@dataclass(frozen=True)
class PaddedByteBatch:
    """Rectangular byte IDs, lengths, and right-padding masks."""

    input_ids: tuple[tuple[int, ...], ...]
    lengths: tuple[int, ...]
    attention_mask: tuple[tuple[int, ...], ...]


def _validate_byte_limit(max_bytes: int | None) -> None:
    if max_bytes is not None and (
        isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0
    ):
        raise ValueError("max_bytes must be a non-negative integer or None.")


def encode_utf8_bytes(
    text: object,
    *,
    max_bytes: int | None = None,
    allow_truncation: bool = False,
    source_start: int = 0,
    sample_id: SampleId = None,
) -> ByteEncoding:
    """Encode exact source text as UTF-8 byte IDs, optionally cropping safely.

    UTF-8 bytes are represented as ``byte_value + 1`` (IDs 1 through 256); ID
    zero remains reserved for padding. Strict overflow is the default. When
    ``allow_truncation`` is true, only complete Unicode code points are kept and
    the omitted half-open source range is returned in ``omitted_char_range``.

    ``source_start`` is the Python-string index of this span in its parent text.
    """
    _validate_byte_limit(max_bytes)
    if isinstance(source_start, bool) or not isinstance(source_start, int) or source_start < 0:
        raise ValueError("source_start must be a non-negative integer.")

    checked = validate_text(text, sample_id=sample_id)
    encoded = checked.encode("utf-8", errors="strict")
    represented_chars = len(checked)

    if max_bytes is not None and len(encoded) > max_bytes:
        if not allow_truncation:
            label = "sample <unspecified>" if sample_id is None else f"sample {sample_id!r}"
            raise ByteSpanOverflowError(
                f"{label} byte span [{source_start}, {source_start + len(checked)}) "
                f"encodes to {len(encoded)} bytes, exceeding max_bytes={max_bytes}; "
                "increase the limit or explicitly enable truncation."
            )

        byte_count = 0
        represented_chars = 0
        for character in checked:
            character_size = len(character.encode("utf-8", errors="strict"))
            if byte_count + character_size > max_bytes:
                break
            byte_count += character_size
            represented_chars += 1
        encoded = encoded[:byte_count]

    omitted_range = None
    if represented_chars < len(checked):
        omitted_range = (source_start + represented_chars, source_start + len(checked))

    ids = tuple(byte_value + BYTE_ID_OFFSET for byte_value in encoded)
    return ByteEncoding(
        ids=ids,
        length=len(ids),
        source_start=source_start,
        source_end=source_start + len(checked),
        represented_char_end=source_start + represented_chars,
        omitted_char_range=omitted_range,
    )


def pad_byte_sequences(
    sequences: Iterable[ByteEncoding | Sequence[int]],
    *,
    pad_to_length: int | None = None,
) -> PaddedByteBatch:
    """Right-pad byte ID sequences with PAD ID 0 and return lengths/masks.

    If ``pad_to_length`` is omitted, the longest sequence determines the width.
    A requested width shorter than any input raises instead of truncating.
    """
    rows = []
    for sequence_index, sequence in enumerate(sequences):
        values = sequence.ids if isinstance(sequence, ByteEncoding) else tuple(sequence)
        for position, value in enumerate(values):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < BYTE_VOCAB_SIZE
            ):
                raise ValueError(
                    f"sequence {sequence_index} contains invalid byte ID {value!r} "
                    f"at position {position}; expected an integer in [0, 256]."
                )
            if value == PAD_ID:
                raise ValueError(
                    f"sequence {sequence_index} contains PAD ID inside unpadded data "
                    f"at position {position}."
                )
        rows.append(tuple(values))

    maximum = max((len(row) for row in rows), default=0)
    if pad_to_length is None:
        width = maximum
    else:
        if (
            isinstance(pad_to_length, bool)
            or not isinstance(pad_to_length, int)
            or pad_to_length < 0
        ):
            raise ValueError("pad_to_length must be a non-negative integer or None.")
        if pad_to_length < maximum:
            raise ValueError(
                f"pad_to_length={pad_to_length} is shorter than the longest input "
                f"sequence ({maximum}); padding never truncates."
            )
        width = pad_to_length

    padded_rows = tuple(row + (PAD_ID,) * (width - len(row)) for row in rows)
    masks = tuple(
        (1,) * len(row) + (0,) * (width - len(row))
        for row in rows
    )
    return PaddedByteBatch(
        input_ids=padded_rows,
        lengths=tuple(len(row) for row in rows),
        attention_mask=masks,
    )
