"""Validation policy for exact, UTF-8-encodable source strings.

This module deliberately does not normalize or rewrite text. The returned
string is the exact input object supplied by the caller.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

SampleId: TypeAlias = str | int | None


class UnicodeInputError(ValueError):
    """Raised when text cannot be represented by the raw UTF-8 pathway."""


def _sample_label(sample_id: SampleId) -> str:
    return "sample <unspecified>" if sample_id is None else f"sample {sample_id!r}"


def validate_text(text: object, *, sample_id: SampleId = None) -> str:
    """Validate a string without normalization, trimming, or character loss.

    Args:
        text: The original input. It must be a Python ``str``.
        sample_id: Optional identifier included in actionable errors.

    Returns:
        The same string value, unchanged.

    Raises:
        TypeError: If ``text`` is not a string.
        UnicodeInputError: If it contains a surrogate code point or cannot be
            encoded using strict UTF-8.
    """
    label = _sample_label(sample_id)
    if not isinstance(text, str):
        raise TypeError(f"{label} must be str; received {type(text).__name__}.")

    for index, character in enumerate(text):
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            raise UnicodeInputError(
                f"{label} contains surrogate U+{codepoint:04X} at character "
                f"index {index}; strict UTF-8 requires Unicode scalar values."
            )

    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise UnicodeInputError(
            f"{label} is not strictly UTF-8 encodable at character index "
            f"{exc.start}: {exc.reason}."
        ) from exc

    return text


def validate_texts(
    texts: Sequence[object],
    *,
    sample_ids: Sequence[SampleId] | None = None,
) -> tuple[str, ...]:
    """Validate a batch of strings and return them unchanged in input order.

    If supplied, ``sample_ids`` must have exactly one ID per text. A single
    string is not accepted as a batch, avoiding accidental character iteration.
    """
    if isinstance(texts, str) or not isinstance(texts, Sequence):
        raise TypeError("texts must be a sequence of strings, not a single string.")
    if sample_ids is not None and len(sample_ids) != len(texts):
        raise ValueError(
            f"sample_ids has length {len(sample_ids)} but texts has length {len(texts)}."
        )

    checked = []
    for index, text in enumerate(texts):
        sample_id = sample_ids[index] if sample_ids is not None else index
        checked.append(validate_text(text, sample_id=sample_id))
    return tuple(checked)
