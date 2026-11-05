"""Shared masked CNN encoder for UTF-8 byte spans.

Input IDs follow :mod:`src.byte_encoding`: zero is padding and byte ``b`` is
represented by ``b + 1``. The encoder accepts packed spans with explicit
lengths and optionally validates a caller-provided right-padding mask.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.byte_encoding import BYTE_VOCAB_SIZE, PAD_ID


class ByteEncoderInputError(ValueError):
    """Raised when byte IDs, lengths, or masks violate the encoder contract."""


class ByteSequenceOverflowError(ValueError):
    """Raised when a real span exceeds a configured byte limit."""


class ByteCNNEncoder(nn.Module):
    """Encode a batch of token/gap byte spans with one shared CNN.

    Args:
        embedding_dim: Width of each byte embedding.
        num_filters: Number of filters for each convolution kernel.
        kernel_sizes: Positive convolution widths.
        output_dim: Width of each final span representation.
        max_span_bytes: Optional strict cap on real (non-padding) span lengths.

    Shapes:
        ``input_ids``: ``[batch, spans, padded_bytes]`` integer IDs.
        ``lengths``: ``[batch, spans]`` integer lengths.
        ``attention_mask``: optional ``[batch, spans, padded_bytes]`` 0/1 mask.
        Return: ``[batch, spans, output_dim]``.

    CNN windows are masked by whether they overlap a real byte. Right-padding
    therefore cannot extend max pooling, and empty spans are forced to exact
    zero after the biased projection layer.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 32,
        num_filters: int = 48,
        kernel_sizes: Sequence[int] = (2, 3, 4),
        output_dim: int = 128,
        max_span_bytes: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer.")
        if isinstance(num_filters, bool) or not isinstance(num_filters, int) or num_filters <= 0:
            raise ValueError("num_filters must be a positive integer.")
        if isinstance(output_dim, bool) or not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError("output_dim must be a positive integer.")
        kernels = tuple(kernel_sizes)
        if not kernels or any(
            isinstance(kernel, bool) or not isinstance(kernel, int) or kernel <= 0
            for kernel in kernels
        ):
            raise ValueError("kernel_sizes must contain positive integers.")
        if max_span_bytes is not None and (
            isinstance(max_span_bytes, bool)
            or not isinstance(max_span_bytes, int)
            or max_span_bytes < 0
        ):
            raise ValueError("max_span_bytes must be a non-negative integer or None.")

        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernels
        self.output_dim = output_dim
        self.max_span_bytes = max_span_bytes
        self.byte_embedding = nn.Embedding(
            BYTE_VOCAB_SIZE,
            embedding_dim,
            padding_idx=PAD_ID,
        )
        self.convolutions = nn.ModuleList(
            nn.Conv1d(
                embedding_dim,
                num_filters,
                kernel_size=kernel,
                padding=kernel - 1,
            )
            for kernel in kernels
        )
        self.projection = nn.Linear(num_filters * len(kernels), output_dim)

    @staticmethod
    def _integer_dtype(dtype: torch.dtype) -> bool:
        return dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)

    def forward(
        self,
        input_ids: Tensor,
        lengths: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if not isinstance(input_ids, Tensor) or input_ids.ndim != 3:
            raise ByteEncoderInputError("input_ids must be a rank-3 tensor [batch, spans, bytes].")
        batch_size, span_count, byte_width = input_ids.shape
        if batch_size == 0 or span_count == 0 or byte_width == 0:
            raise ByteEncoderInputError("input_ids dimensions must all be non-zero.")
        if not self._integer_dtype(input_ids.dtype):
            raise ByteEncoderInputError("input_ids must use an integer tensor dtype.")
        if input_ids.device != self.byte_embedding.weight.device:
            raise ByteEncoderInputError("input_ids must be on the same device as the byte encoder.")
        if torch.any(input_ids < PAD_ID) or torch.any(input_ids >= BYTE_VOCAB_SIZE):
            raise ByteEncoderInputError("byte IDs must be in [0, 256].")

        if not isinstance(lengths, Tensor) or lengths.ndim != 2:
            raise ByteEncoderInputError("lengths must be a rank-2 tensor [batch, spans].")
        if lengths.shape != (batch_size, span_count):
            raise ByteEncoderInputError(
                f"lengths shape {tuple(lengths.shape)} does not match "
                f"input batch/spans {(batch_size, span_count)}."
            )
        if not self._integer_dtype(lengths.dtype) or lengths.dtype == torch.bool:
            raise ByteEncoderInputError("lengths must use an integer tensor dtype.")
        if lengths.device != input_ids.device:
            raise ByteEncoderInputError("lengths and input_ids must be on the same device.")
        if torch.any(lengths < 0) or torch.any(lengths > byte_width):
            raise ByteEncoderInputError(f"lengths must be in [0, {byte_width}].")
        if self.max_span_bytes is not None and torch.any(lengths > self.max_span_bytes):
            observed = int(lengths.max().item())
            raise ByteSequenceOverflowError(
                f"a byte span has length {observed}, exceeding max_span_bytes="
                f"{self.max_span_bytes}; increase the cap or preprocess explicitly."
            )

        positions = torch.arange(byte_width, device=input_ids.device)
        expected_mask = positions.view(1, 1, byte_width) < lengths.unsqueeze(-1)
        if attention_mask is not None:
            if not isinstance(attention_mask, Tensor) or attention_mask.shape != input_ids.shape:
                raise ByteEncoderInputError("attention_mask must have the same shape as input_ids.")
            if attention_mask.device != input_ids.device:
                raise ByteEncoderInputError("attention_mask and input_ids must be on the same device.")
            if attention_mask.dtype != torch.bool and not self._integer_dtype(attention_mask.dtype):
                raise ByteEncoderInputError("attention_mask values must use a boolean or integer dtype.")
            if torch.any((attention_mask != 0) & (attention_mask != 1)):
                raise ByteEncoderInputError("attention_mask values must be 0 or 1.")
            if not torch.equal(attention_mask.bool(), expected_mask):
                raise ByteEncoderInputError(
                    "attention_mask must identify exactly the right-padded positions given lengths."
                )

        if torch.any(input_ids.masked_select(expected_mask) == PAD_ID):
            raise ByteEncoderInputError("PAD ID 0 cannot occur inside a declared byte span.")
        if torch.any(input_ids.masked_select(~expected_mask) != PAD_ID):
            raise ByteEncoderInputError("positions after each declared length must contain PAD ID 0.")

        embedded = self.byte_embedding(input_ids.long()).reshape(
            batch_size * span_count, byte_width, self.embedding_dim
        ).transpose(1, 2)
        flat_lengths = lengths.reshape(-1)
        pooled_features = []
        for kernel, convolution in zip(self.kernel_sizes, self.convolutions):
            activated = F.relu(convolution(embedded))
            output_width = activated.shape[-1]
            # With padding=kernel-1, output index j sees real input for
            # 0 <= j < length + kernel - 1.
            valid_widths = flat_lengths + kernel - 1
            output_positions = torch.arange(output_width, device=input_ids.device)
            valid = output_positions.unsqueeze(0) < valid_widths.unsqueeze(1)
            pooled = activated.masked_fill(~valid.unsqueeze(1), torch.finfo(activated.dtype).min).amax(dim=2)
            pooled = torch.where(flat_lengths.unsqueeze(1) > 0, pooled, torch.zeros_like(pooled))
            pooled_features.append(pooled)

        merged = torch.cat(pooled_features, dim=1)
        encoded = F.relu(self.projection(merged))
        encoded = torch.where(flat_lengths.unsqueeze(1) > 0, encoded, torch.zeros_like(encoded))
        return encoded.reshape(batch_size, span_count, self.output_dim)
