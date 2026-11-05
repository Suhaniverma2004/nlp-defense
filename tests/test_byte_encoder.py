import pytest
import torch

from src.byte_encoder import (
    ByteCNNEncoder,
    ByteEncoderInputError,
    ByteSequenceOverflowError,
)
from src.byte_encoding import BYTE_VOCAB_SIZE, PAD_ID, encode_utf8_bytes


def small_encoder(**kwargs):
    return ByteCNNEncoder(
        embedding_dim=8,
        num_filters=4,
        kernel_sizes=(2, 3),
        output_dim=6,
        **kwargs,
    )


def test_byte_id_mapping_and_encoder_output_shape():
    encoded = encode_utf8_bytes("Aé")
    assert encoded.ids == tuple(value + 1 for value in b"A\xc3\xa9")
    assert min(encoded.ids) >= 1
    assert max(encoded.ids) <= 256

    encoder = small_encoder()
    ids = torch.tensor([[[66, 196, 170], [0, 0, 0]]], dtype=torch.long)
    lengths = torch.tensor([[3, 0]], dtype=torch.long)
    mask = torch.tensor([[[1, 1, 1], [0, 0, 0]]], dtype=torch.bool)
    output = encoder(ids, lengths, mask)

    assert encoder.byte_embedding.num_embeddings == BYTE_VOCAB_SIZE == 257
    assert encoder.byte_embedding.padding_idx == PAD_ID == 0
    assert output.shape == (1, 2, 6)


def test_masked_pooling_is_invariant_to_extra_right_padding():
    torch.manual_seed(4)
    encoder = small_encoder().eval()
    ids_short = torch.tensor([[[67, 98]]], dtype=torch.long)
    ids_long = torch.tensor([[[67, 98, 0, 0, 0]]], dtype=torch.long)
    lengths = torch.tensor([[2]], dtype=torch.long)

    short = encoder(ids_short, lengths)
    long = encoder(ids_long, lengths)

    torch.testing.assert_close(short, long, rtol=0, atol=0)


def test_empty_spans_are_exact_zero_despite_convolution_and_projection_biases():
    encoder = small_encoder().eval()
    with torch.no_grad():
        for convolution in encoder.convolutions:
            convolution.bias.fill_(2.5)
        encoder.projection.bias.fill_(3.5)
    ids = torch.zeros((2, 3, 2), dtype=torch.long)
    lengths = torch.zeros((2, 3), dtype=torch.long)

    output = encoder(ids, lengths)

    assert torch.equal(output, torch.zeros_like(output))


def test_variable_span_lengths_batch_and_gradient_flow():
    torch.manual_seed(9)
    encoder = small_encoder()
    ids = torch.tensor(
        [
            [[66, 0, 0], [196, 170, 0]],
            [[1, 2, 3], [0, 0, 0]],
        ],
        dtype=torch.long,
    )
    lengths = torch.tensor([[1, 2], [3, 0]], dtype=torch.long)
    mask = torch.arange(3).view(1, 1, 3) < lengths.unsqueeze(-1)
    output = encoder(ids, lengths, mask)
    output.sum().backward()

    assert output.shape == (2, 2, 6)
    assert torch.isfinite(output).all()
    assert encoder.byte_embedding.weight.grad is not None
    assert encoder.byte_embedding.weight.grad.abs().sum() > 0
    assert encoder.convolutions[0].weight.grad is not None
    assert encoder.convolutions[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "ids,lengths,mask,message",
    [
        (torch.ones((1, 2), dtype=torch.long), torch.ones((1, 1), dtype=torch.long), None, "rank-3"),
        (torch.tensor([[[257]]]), torch.tensor([[1]]), None, r"\[0, 256\]"),
        (torch.tensor([[[0]]]), torch.tensor([[1]]), None, "PAD ID"),
        (torch.tensor([[[1, 0]]]), torch.tensor([[1]]), torch.tensor([[[1, 1]]]), "right-padded"),
        (torch.tensor([[[1, 0]]]), torch.tensor([[3]]), None, "lengths must be"),
    ],
)
def test_invalid_ids_shapes_lengths_and_masks_are_rejected(ids, lengths, mask, message):
    with pytest.raises(ByteEncoderInputError, match=message):
        small_encoder()(ids, lengths, mask)


def test_configured_span_overflow_is_rejected_without_truncation():
    encoder = small_encoder(max_span_bytes=2)
    ids = torch.tensor([[[1, 2, 3]]], dtype=torch.long)
    lengths = torch.tensor([[3]], dtype=torch.long)

    with pytest.raises(ByteSequenceOverflowError, match="max_span_bytes=2"):
        encoder(ids, lengths)


def test_invalid_encoder_configuration_is_rejected():
    with pytest.raises(ValueError, match="kernel_sizes"):
        ByteCNNEncoder(kernel_sizes=())
