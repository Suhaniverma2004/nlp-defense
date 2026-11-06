"""Canonical byte-aware DistilBERT sentiment model.

The input boundary is :class:`src.alignment.AlignmentResult`; tokenization and
Unicode preprocessing are intentionally not reimplemented here. Legacy
character-vocabulary model APIs remain available in their original modules and
are not aliases for this byte-based architecture.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from transformers import DistilBertModel

from src.alignment import (
    AlignmentResult,
    AlignmentUnit,
    pad_alignment_units,
)
from src.byte_encoder import ByteCNNEncoder
from src.byte_encoding import PAD_ID

UNIT_PADDING = 0
UNIT_TOKEN = 1
UNIT_GAP = 2
UNIT_SPECIAL = 3


_COLLATOR_PROVENANCE = object()


@dataclass(frozen=True)
class AlignedModelBatch:
    """Padded tensors derived only from Milestone 2.1 alignment results.

    ``input_ids`` contains WordPiece IDs for token/special units and the model's
    PAD ID as a placeholder for gaps and padding. It is used only for the word
    embedding lookup; the full model is called through ``inputs_embeds``.
    """

    input_ids: Tensor
    byte_input_ids: Tensor
    byte_lengths: Tensor
    byte_attention_mask: Tensor
    unit_type_ids: Tensor
    attention_mask: Tensor
    source_texts: tuple[str, ...]
    unit_offsets: tuple[tuple[tuple[int, int], ...], ...]
    _alignment_provenance: object | None = field(default=None, init=False, repr=False, compare=False)

    def __reduce__(self):
        """Serialize tensor data through a validating reconstruction path.

        Pickle does not carry a caller-provided provenance marker. The restore
        helper rechecks ordered source coverage and byte ownership before it
        grants the private in-process marker.
        """
        return (
            _restore_aligned_model_batch,
            (
                self.input_ids,
                self.byte_input_ids,
                self.byte_lengths,
                self.byte_attention_mask,
                self.unit_type_ids,
                self.attention_mask,
                self.source_texts,
                self.unit_offsets,
            ),
        )

    def to(self, device: torch.device | str) -> AlignedModelBatch:
        """Return a copy with every tensor moved to ``device``."""
        return _restore_aligned_model_batch(
            self.input_ids.to(device),
            self.byte_input_ids.to(device),
            self.byte_lengths.to(device),
            self.byte_attention_mask.to(device),
            self.unit_type_ids.to(device),
            self.attention_mask.to(device),
            self.source_texts,
            self.unit_offsets,
        )


def _restore_aligned_model_batch(
    input_ids: Tensor,
    byte_input_ids: Tensor,
    byte_lengths: Tensor,
    byte_attention_mask: Tensor,
    unit_type_ids: Tensor,
    attention_mask: Tensor,
    source_texts: tuple[str, ...],
    unit_offsets: tuple[tuple[tuple[int, int], ...], ...],
) -> AlignedModelBatch:
    """Revalidate serialized batch ownership before restoring provenance."""
    if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
        raise ValueError("serialized aligned batch input_ids must have shape [batch, units].")
    batch_size, unit_count = input_ids.shape
    if not isinstance(byte_input_ids, Tensor) or byte_input_ids.ndim != 3 or byte_input_ids.shape[-1] == 0:
        raise ValueError("serialized aligned batch byte_input_ids must have shape [batch, units, bytes].")
    expected_shapes = (
        (byte_lengths, (batch_size, unit_count), "byte_lengths"),
        (unit_type_ids, (batch_size, unit_count), "unit_type_ids"),
        (attention_mask, (batch_size, unit_count), "attention_mask"),
    )
    for value, expected, name in expected_shapes:
        if not isinstance(value, Tensor) or tuple(value.shape) != expected:
            raise ValueError(f"serialized aligned batch {name} has an invalid shape.")
    if byte_input_ids.shape[:2] != (batch_size, unit_count):
        raise ValueError("serialized aligned batch byte_input_ids has an invalid shape.")
    if not isinstance(byte_attention_mask, Tensor) or byte_attention_mask.shape != byte_input_ids.shape:
        raise ValueError("serialized aligned batch byte_attention_mask has an invalid shape.")
    if batch_size == 0 or unit_count == 0 or len(source_texts) != batch_size or len(unit_offsets) != batch_size:
        raise ValueError("serialized aligned batch metadata does not match its tensor dimensions.")
    if any(not isinstance(text, str) for text in source_texts):
        raise ValueError("serialized aligned batch source_texts must contain strings.")
    if any(len(row) != unit_count for row in unit_offsets):
        raise ValueError("serialized aligned batch offsets do not match its unit dimensions.")
    tensors = (byte_input_ids, byte_lengths, byte_attention_mask, unit_type_ids, attention_mask)
    if any(value.device != input_ids.device for value in tensors):
        raise ValueError("serialized aligned batch tensors must share one device.")
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if any(value.dtype not in integer_dtypes for value in (input_ids, byte_input_ids, byte_lengths, unit_type_ids)):
        raise ValueError("serialized aligned batch IDs, lengths, and types must be integer tensors.")
    if attention_mask.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("serialized aligned batch attention mask must be boolean or integer.")
    if byte_attention_mask.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("serialized aligned batch byte mask must be boolean or integer.")
    if torch.any((attention_mask != 0) & (attention_mask != 1)):
        raise ValueError("serialized aligned batch attention mask must contain only zero or one.")
    if torch.any((unit_type_ids < UNIT_PADDING) | (unit_type_ids > UNIT_SPECIAL)):
        raise ValueError("serialized aligned batch contains an unknown unit type.")
    if torch.any(byte_lengths < 0) or torch.any(byte_lengths > byte_input_ids.shape[-1]):
        raise ValueError("serialized aligned batch contains an invalid byte length.")
    if torch.any((byte_attention_mask != 0) & (byte_attention_mask != 1)):
        raise ValueError("serialized aligned batch byte mask must contain only zero or one.")

    batch = AlignedModelBatch(
        input_ids=input_ids,
        byte_input_ids=byte_input_ids,
        byte_lengths=byte_lengths,
        byte_attention_mask=byte_attention_mask,
        unit_type_ids=unit_type_ids,
        attention_mask=attention_mask,
        source_texts=source_texts,
        unit_offsets=unit_offsets,
    )
    for row_index, (text, offsets) in enumerate(zip(source_texts, unit_offsets)):
        cursor = 0
        saw_padding = False
        for unit_index, raw_pair in enumerate(offsets):
            if not isinstance(raw_pair, (tuple, list)) or len(raw_pair) != 2:
                raise ValueError("serialized aligned batch contains a malformed source offset.")
            start, end = raw_pair
            if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end)):
                raise ValueError("serialized aligned batch offsets must be integer character ranges.")
            unit_type = int(unit_type_ids[row_index, unit_index].item())
            attended = bool(attention_mask[row_index, unit_index].item())
            length = int(byte_lengths[row_index, unit_index].item())
            if unit_type == UNIT_PADDING:
                saw_padding = True
                if (
                    attended
                    or (start, end) != (0, 0)
                    or length != 0
                    or torch.any(byte_input_ids[row_index, unit_index] != 0)
                    or torch.any(byte_attention_mask[row_index, unit_index])
                ):
                    raise ValueError("serialized aligned padding must own no source or bytes.")
                continue
            if saw_padding:
                raise ValueError("serialized aligned units cannot follow sequence padding.")
            if not attended:
                raise ValueError("serialized source and special units must be attended.")
            if unit_type == UNIT_SPECIAL:
                if (
                    (start, end) != (0, 0)
                    or length != 0
                    or torch.any(byte_input_ids[row_index, unit_index] != 0)
                    or torch.any(byte_attention_mask[row_index, unit_index])
                ):
                    raise ValueError("serialized structural special units cannot own source bytes.")
                continue
            if unit_type not in (UNIT_TOKEN, UNIT_GAP):
                raise ValueError("serialized aligned batch contains an unknown unit type.")
            if unit_type == UNIT_TOKEN and start == end == cursor and length == 0:
                if torch.any(byte_input_ids[row_index, unit_index] != 0) or torch.any(
                    byte_attention_mask[row_index, unit_index]
                ):
                    raise ValueError("serialized zero-length token units cannot carry bytes.")
                continue
            if start != cursor or end <= start or end > len(text):
                raise ValueError("serialized source units do not form an ordered exact partition.")
            if length <= 0 or length > byte_input_ids.shape[-1]:
                raise ValueError("serialized source unit has an invalid byte length.")
            ids = byte_input_ids[row_index, unit_index]
            if (
                torch.any(ids[:length] < 1)
                or torch.any(ids[:length] > 256)
                or torch.any(ids[length:] != 0)
            ):
                raise ValueError("serialized source unit has invalid byte IDs or padding.")
            mask = byte_attention_mask[row_index, unit_index].bool()
            expected_mask = torch.arange(ids.shape[0], device=ids.device) < length
            if not torch.equal(mask, expected_mask):
                raise ValueError("serialized source unit byte mask disagrees with its length.")
            raw_bytes = bytes(int(value) - 1 for value in ids[:length].tolist())
            try:
                encoded_piece = text[start:end].encode("utf-8", errors="strict")
            except UnicodeEncodeError as exc:
                raise ValueError("serialized source text contains an unpaired surrogate.") from exc
            if raw_bytes != encoded_piece:
                raise ValueError("serialized byte span does not match its original source range.")
            cursor = end
        if cursor != len(text):
            raise ValueError(f"serialized aligned row {row_index} does not cover its full source text.")

    object.__setattr__(batch, "_alignment_provenance", _COLLATOR_PROVENANCE)
    return batch


@dataclass(frozen=True)
class CanonicalClassifierOutput:
    """Model outputs; calculate training loss separately from these logits."""

    logits: Tensor
    last_hidden_state: Tensor
    attention_mask: Tensor


def _kind_id(unit: AlignmentUnit) -> int:
    if unit.kind == "token":
        return UNIT_TOKEN
    if unit.kind == "gap":
        return UNIT_GAP
    if unit.kind == "special":
        return UNIT_SPECIAL
    if unit.kind == "padding":
        return UNIT_PADDING
    raise ValueError(f"unsupported alignment unit kind: {unit.kind!r}.")


def collate_alignment_results(
    alignments: Sequence[AlignmentResult],
    *,
    pad_token_id: int = PAD_ID,
) -> AlignedModelBatch:
    """Pad alignment results into model tensors without truncating source data.

    Input order is retained. Gap units keep their original positions and bytes;
    sequence padding is represented separately with a zero attention mask.
    Truncated or incompletely covered alignments are rejected rather than
    combining their shortened units with the full tokenizer result.

    The returned batch carries private collator provenance. Constructing an
    ``AlignedModelBatch`` directly is unsupported; ``forward`` rejects batches
    that did not come from this collator. This avoids reimplementing the
    Milestone 2.1 source-partition and byte-coverage checks here.
    """
    if isinstance(alignments, (str, bytes)) or not isinstance(alignments, Sequence):
        raise TypeError("alignments must be a sequence of AlignmentResult objects.")
    if not alignments:
        raise ValueError("at least one alignment is required to build a model batch.")
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int) or pad_token_id < 0:
        raise ValueError("pad_token_id must be a non-negative integer.")

    for index, result in enumerate(alignments):
        if not isinstance(result, AlignmentResult):
            raise TypeError(f"alignment {index} is not an AlignmentResult.")
        coverage = result.coverage
        if (
            not coverage.fully_represented
            or coverage.token_truncated
            or coverage.expanded_truncated
            or coverage.byte_truncated
        ):
            raise ValueError(
                f"alignment {index} has incomplete or truncated source coverage; "
                "the canonical model requires complete, untruncated Milestone 2.1 alignments."
            )
        if not result.units:
            raise ValueError(f"alignment {index} has no model units and cannot be classified.")

    padded = pad_alignment_units(alignments, pad_token_id=pad_token_id)
    batch_size = len(padded.units)
    unit_count = len(padded.units[0])
    if unit_count == 0:
        raise ValueError("alignment batch has no expanded units.")
    max_bytes = max(
        1,
        max(
            (unit.byte_length for row in padded.units for unit in row),
            default=0,
        ),
    )

    token_ids = torch.full((batch_size, unit_count), pad_token_id, dtype=torch.long)
    byte_ids = torch.zeros((batch_size, unit_count, max_bytes), dtype=torch.long)
    byte_lengths = torch.zeros((batch_size, unit_count), dtype=torch.long)
    unit_types = torch.zeros((batch_size, unit_count), dtype=torch.long)
    unit_offsets: list[tuple[tuple[int, int], ...]] = []

    for batch_index, row in enumerate(padded.units):
        offsets_for_row = []
        for unit_index, unit in enumerate(row):
            unit_types[batch_index, unit_index] = _kind_id(unit)
            offsets_for_row.append((unit.source_start, unit.source_end))
            if unit.kind in ("token", "special"):
                if unit.token_id is None:
                    raise ValueError(
                        f"alignment {batch_index}, unit {unit_index} has no token ID."
                    )
                token_ids[batch_index, unit_index] = unit.token_id
            elif unit.kind == "gap":
                if unit.token_id is not None:
                    raise ValueError(
                        f"alignment {batch_index}, unit {unit_index} unexpectedly has a token ID."
                    )
            elif unit.kind == "padding" and unit.token_id != pad_token_id:
                raise ValueError(
                    f"alignment {batch_index}, padding unit {unit_index} does not use pad_token_id."
                )

            if unit.byte_length != len(unit.byte_ids):
                raise ValueError(
                    f"alignment {batch_index}, unit {unit_index} byte length does not match byte IDs."
                )
            if unit.kind in ("special", "padding") and unit.byte_length != 0:
                raise ValueError(
                    f"alignment {batch_index}, {unit.kind} unit {unit_index} cannot own bytes."
                )
            if unit.kind == "gap" and unit.byte_length == 0:
                raise ValueError(
                    f"alignment {batch_index}, gap unit {unit_index} must contain source bytes."
                )
            if unit.byte_length:
                byte_ids[batch_index, unit_index, : unit.byte_length] = torch.tensor(
                    unit.byte_ids,
                    dtype=torch.long,
                )
                byte_lengths[batch_index, unit_index] = unit.byte_length
        unit_offsets.append(tuple(offsets_for_row))

    byte_mask = (
        torch.arange(max_bytes).view(1, 1, max_bytes)
        < byte_lengths.unsqueeze(-1)
    )
    batch = AlignedModelBatch(
        input_ids=token_ids,
        byte_input_ids=byte_ids,
        byte_lengths=byte_lengths,
        byte_attention_mask=byte_mask,
        unit_type_ids=unit_types,
        attention_mask=torch.tensor(padded.attention_mask, dtype=torch.long),
        source_texts=tuple(result.text for result in alignments),
        unit_offsets=tuple(unit_offsets),
    )
    object.__setattr__(batch, "_alignment_provenance", _COLLATOR_PROVENANCE)
    return batch


class CanonicalByteAwareDistilBert(nn.Module):
    """DistilBERT plus a shared UTF-8 byte CNN for SST-2 logits.

    A caller supplies an already-created :class:`DistilBertModel`; this class
    does not load a checkpoint or tokenizer. Tests can pass a tiny local config
    model, and applications can explicitly supply their selected backbone.
    The tokenizer and raw-text alignment remain caller-owned and explicit.

    Forward input must be an :class:`AlignedModelBatch` produced by
    :func:`collate_alignment_results`; directly constructed batches are
    rejected because their source ownership/byte coverage has not been
    established by the Milestone 2.1 alignment path. Output logits have shape
    ``[batch, 2]``. Cross-entropy loss is intentionally computed by the caller.

    ``fusion_gate`` is a trainable scalar parameter. Its sigmoid produces one
    gate value shared across all hidden dimensions and all content-token
    positions, controlling the magnitude of the projected byte residual.
    """

    def __init__(
        self,
        backbone: DistilBertModel,
        *,
        num_labels: int = 2,
        byte_embedding_dim: int = 32,
        byte_num_filters: int = 48,
        byte_kernel_sizes: Sequence[int] = (2, 3, 4),
        byte_output_dim: int = 128,
        max_span_bytes: int | None = None,
        dropout: float = 0.1,
        pad_token_id: int = PAD_ID,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, DistilBertModel):
            raise TypeError("backbone must be an explicitly constructed DistilBertModel.")
        if isinstance(num_labels, bool) or not isinstance(num_labels, int) or num_labels != 2:
            raise ValueError("the initial SST-2 classifier requires exactly two labels.")
        if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int) or pad_token_id < 0:
            raise ValueError("pad_token_id must be a non-negative integer.")
        if pad_token_id >= backbone.config.vocab_size:
            raise ValueError("pad_token_id is outside the DistilBERT vocabulary.")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0 <= dropout < 1:
            raise ValueError("dropout must be a number in [0, 1).")

        self.backbone = backbone
        self.hidden_size = int(backbone.config.dim)
        self.max_position_embeddings = int(backbone.config.max_position_embeddings)
        self.pad_token_id = pad_token_id
        self.byte_encoder = ByteCNNEncoder(
            embedding_dim=byte_embedding_dim,
            num_filters=byte_num_filters,
            kernel_sizes=byte_kernel_sizes,
            output_dim=byte_output_dim,
            max_span_bytes=max_span_bytes,
        )
        self.token_byte_projection = nn.Linear(byte_output_dim, self.hidden_size)
        self.gap_byte_projection = nn.Linear(byte_output_dim, self.hidden_size)
        # One learned sigmoid gate is shared over channels and token positions.
        self.fusion_gate = nn.Parameter(torch.zeros(()))
        self.gap_type_embedding = nn.Parameter(torch.empty(self.hidden_size))
        nn.init.normal_(self.gap_type_embedding, mean=0.0, std=0.02)
        self.classifier_dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def _validate_batch(self, batch: AlignedModelBatch) -> None:
        if not isinstance(batch, AlignedModelBatch):
            raise TypeError("forward expects an AlignedModelBatch from collate_alignment_results().")
        if batch._alignment_provenance is not _COLLATOR_PROVENANCE:
            raise ValueError(
                "AlignedModelBatch must be produced by collate_alignment_results(); "
                "direct construction cannot establish source coverage provenance."
            )
        if batch.input_ids.ndim != 2:
            raise ValueError("batch.input_ids must have shape [batch, units].")
        batch_size, unit_count = batch.input_ids.shape
        if batch_size == 0 or unit_count == 0:
            raise ValueError("model batches must contain at least one example and one unit.")
        if unit_count > self.max_position_embeddings:
            raise ValueError(
                f"expanded sequence length {unit_count} exceeds DistilBERT's position limit "
                f"{self.max_position_embeddings}; shorten inputs explicitly before alignment."
            )
        if batch.byte_input_ids.ndim != 3 or batch.byte_input_ids.shape[:2] != (batch_size, unit_count):
            raise ValueError("byte_input_ids must have shape [batch, units, bytes].")
        if batch.byte_lengths.shape != (batch_size, unit_count):
            raise ValueError("byte_lengths must have shape [batch, units].")
        if batch.byte_attention_mask.shape != batch.byte_input_ids.shape:
            raise ValueError("byte_attention_mask must have the same shape as byte_input_ids.")
        if batch.unit_type_ids.shape != (batch_size, unit_count):
            raise ValueError("unit_type_ids must have shape [batch, units].")
        if batch.attention_mask.shape != (batch_size, unit_count):
            raise ValueError("attention_mask must have shape [batch, units].")
        if len(batch.source_texts) != batch_size or len(batch.unit_offsets) != batch_size:
            raise ValueError("source text and offset metadata must contain one row per example.")
        if any(len(row) != unit_count for row in batch.unit_offsets):
            raise ValueError("each unit_offsets row must match the padded unit count.")
        tensor_devices = {
            batch.input_ids.device,
            batch.byte_input_ids.device,
            batch.byte_lengths.device,
            batch.byte_attention_mask.device,
            batch.unit_type_ids.device,
            batch.attention_mask.device,
        }
        if len(tensor_devices) != 1:
            raise ValueError("all tensors in AlignedModelBatch must be on the same device.")
        if batch.input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must use an integer tensor dtype.")
        if batch.unit_type_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("unit_type_ids must use an integer tensor dtype.")
        if batch.attention_mask.dtype not in (torch.bool, torch.uint8, torch.int32, torch.int64):
            raise ValueError("attention_mask must use a boolean or integer tensor dtype.")
        if torch.any((batch.unit_type_ids < UNIT_PADDING) | (batch.unit_type_ids > UNIT_SPECIAL)):
            raise ValueError("unit_type_ids contains an unknown unit kind.")
        if torch.any((batch.attention_mask != 0) & (batch.attention_mask != 1)):
            raise ValueError("attention_mask values must be 0 or 1.")

        padding_mask = batch.unit_type_ids == UNIT_PADDING
        if not torch.equal(batch.attention_mask.bool(), ~padding_mask):
            raise ValueError("attention_mask must be 1 for source units and 0 only for padding.")
        token_or_special = (batch.unit_type_ids == UNIT_TOKEN) | (batch.unit_type_ids == UNIT_SPECIAL)
        if torch.any(batch.input_ids[token_or_special] < 0) or torch.any(
            batch.input_ids[token_or_special] >= self.backbone.config.vocab_size
        ):
            raise ValueError("token or special ID is outside the DistilBERT vocabulary.")
        if torch.any(batch.input_ids[(~token_or_special)] != self.pad_token_id):
            raise ValueError("gap and padding input IDs must use the configured PAD placeholder.")

        byte_lengths = batch.byte_lengths
        no_byte_units = (batch.unit_type_ids == UNIT_SPECIAL) | padding_mask
        if torch.any(byte_lengths[no_byte_units] != 0):
            raise ValueError("structural special and padding units must not carry byte spans.")
        if torch.any(batch.byte_lengths[batch.unit_type_ids == UNIT_GAP] <= 0):
            raise ValueError("gap units must carry a non-empty raw-byte span.")
        if torch.any(batch.attention_mask.sum(dim=1) == 0):
            raise ValueError("each example must contain at least one attended unit.")

    def forward(self, batch: AlignedModelBatch) -> CanonicalClassifierOutput:
        """Return two-class logits and hidden states without calculating loss."""
        self._validate_batch(batch)
        device = self.backbone.embeddings.word_embeddings.weight.device
        batch = batch.to(device)

        word_embeddings = self.backbone.embeddings.word_embeddings(batch.input_ids)
        byte_features = self.byte_encoder(
            batch.byte_input_ids,
            batch.byte_lengths,
            batch.byte_attention_mask,
        )

        token_mask = batch.unit_type_ids == UNIT_TOKEN
        gap_mask = batch.unit_type_ids == UNIT_GAP
        special_mask = batch.unit_type_ids == UNIT_SPECIAL
        padding_mask = batch.unit_type_ids == UNIT_PADDING

        token_byte_features = self.token_byte_projection(byte_features)
        token_byte_present = token_mask & (batch.byte_lengths > 0)
        token_byte_features = torch.where(
            token_byte_present.unsqueeze(-1),
            token_byte_features,
            torch.zeros_like(token_byte_features),
        )
        token_embeddings = word_embeddings + torch.sigmoid(self.fusion_gate) * token_byte_features

        gap_embeddings = self.gap_type_embedding.view(1, 1, -1) + self.gap_byte_projection(byte_features)
        unit_embeddings = torch.where(token_mask.unsqueeze(-1), token_embeddings, torch.zeros_like(word_embeddings))
        unit_embeddings = torch.where(gap_mask.unsqueeze(-1), gap_embeddings, unit_embeddings)
        unit_embeddings = torch.where(special_mask.unsqueeze(-1), word_embeddings, unit_embeddings)
        unit_embeddings = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(unit_embeddings), unit_embeddings)

        # In Transformers 5.2.0, this call sends inputs_embeds through
        # DistilBertModel.embeddings exactly once. That module adds positions,
        # LayerNorm, and dropout before the transformer blocks.
        backbone_output = self.backbone(
            input_ids=None,
            inputs_embeds=unit_embeddings,
            attention_mask=batch.attention_mask,
            return_dict=True,
        )
        hidden = backbone_output.last_hidden_state
        weights = batch.attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(self.classifier_dropout(pooled))
        return CanonicalClassifierOutput(
            logits=logits,
            last_hidden_state=hidden,
            attention_mask=batch.attention_mask,
        )
