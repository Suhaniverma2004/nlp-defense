from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from transformers import DistilBertConfig, DistilBertModel

from src.alignment import AlignmentError, align_text
from src.canonical_model import (
    AlignedModelBatch,
    CanonicalByteAwareDistilBert,
    UNIT_GAP,
    UNIT_PADDING,
    UNIT_SPECIAL,
    UNIT_TOKEN,
    collate_alignment_results,
)


class FakeFastTokenizer:
    is_fast = True

    def __init__(self, text, spans, ids=None):
        content_ids = ids if ids is not None else list(range(10, 10 + len(spans)))
        self.output = {
            "input_ids": [101, *content_ids, 102],
            "offset_mapping": [(0, 0), *spans, (0, 0)],
            "special_tokens_mask": [1, *([0] * len(spans)), 1],
        }

    def __call__(self, text, **kwargs):
        return self.output


def make_alignment(text, spans, ids=None, **kwargs):
    return align_text(text, FakeFastTokenizer(text, spans, ids), **kwargs)


def tiny_backbone(max_positions=16, return_dict=True):
    torch.manual_seed(14)
    config = DistilBertConfig(
        vocab_size=30522,
        max_position_embeddings=max_positions,
        n_layers=1,
        dim=32,
        hidden_dim=64,
        n_heads=4,
        dropout=0.0,
        attention_dropout=0.0,
        seq_classif_dropout=0.0,
        return_dict=return_dict,
    )
    return DistilBertModel(config)


def tiny_model(max_positions=16):
    return CanonicalByteAwareDistilBert(
        tiny_backbone(max_positions),
        byte_embedding_dim=8,
        byte_num_filters=4,
        byte_kernel_sizes=(2, 3),
        byte_output_dim=12,
        dropout=0.0,
    )


def test_collator_preserves_token_gap_special_order_and_right_padding():
    short = make_alignment("ab", [(0, 2)], ids=[15])
    long = make_alignment("a b", [(0, 1), (2, 3)], ids=[16, 17])

    batch = collate_alignment_results([short, long])

    assert batch.input_ids.shape == (2, 5)
    assert batch.byte_input_ids.shape[:2] == (2, 5)
    assert batch.unit_type_ids.tolist() == [
        [UNIT_SPECIAL, UNIT_TOKEN, UNIT_SPECIAL, UNIT_PADDING, UNIT_PADDING],
        [UNIT_SPECIAL, UNIT_TOKEN, UNIT_GAP, UNIT_TOKEN, UNIT_SPECIAL],
    ]
    assert batch.attention_mask.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    assert batch.input_ids[0, 3:].tolist() == [0, 0]
    assert batch.byte_input_ids[1, 2, : batch.byte_lengths[1, 2]].tolist() == [ord(" ") + 1]
    assert batch.source_texts == ("ab", "a b")
    assert batch.unit_offsets[1] == ((0, 0), (0, 1), (1, 2), (2, 3), (0, 0))


def test_model_preserves_multiple_leading_inter_token_and_trailing_gaps():
    text = "  a\tb  c "
    alignment = make_alignment(text, [(2, 3), (4, 5), (7, 8)], ids=[16, 17, 18])
    batch = collate_alignment_results([alignment])

    expected_offsets = (
        (0, 0),
        (0, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 7),
        (7, 8),
        (8, 9),
        (0, 0),
    )
    expected_kinds = [
        UNIT_SPECIAL,
        UNIT_GAP,
        UNIT_TOKEN,
        UNIT_GAP,
        UNIT_TOKEN,
        UNIT_GAP,
        UNIT_TOKEN,
        UNIT_GAP,
        UNIT_SPECIAL,
    ]
    assert batch.unit_offsets[0] == expected_offsets
    assert batch.unit_type_ids[0].tolist() == expected_kinds

    source_owned_indices = [
        index
        for index, kind in enumerate(expected_kinds)
        if kind in (UNIT_TOKEN, UNIT_GAP)
    ]
    owned_offsets = [batch.unit_offsets[0][index] for index in source_owned_indices]
    assert owned_offsets == [(0, 2), (2, 3), (3, 4), (4, 5), (5, 7), (7, 8), (8, 9)]
    reconstructed_text = "".join(text[start:end] for start, end in owned_offsets)
    reconstructed_bytes = b"".join(
        bytes(
            byte_id - 1
            for byte_id in batch.byte_input_ids[0, index, : batch.byte_lengths[0, index]]
        )
        for index in source_owned_indices
    )
    assert reconstructed_text == text
    assert reconstructed_bytes == text.encode("utf-8")
    assert len(owned_offsets) == len(set(owned_offsets))

    output = tiny_model().eval()(batch)
    assert output.logits.shape == (1, 2)
    assert torch.isfinite(output.logits).all()


def test_empty_original_text_is_classifiable_with_structural_units_only():
    alignment = make_alignment("", [])
    batch = collate_alignment_results([alignment])

    assert batch.source_texts == ("",)
    assert batch.unit_type_ids.tolist() == [[UNIT_SPECIAL, UNIT_SPECIAL]]
    assert batch.byte_lengths.tolist() == [[0, 0]]
    output = tiny_model().eval()(batch)
    assert output.logits.shape == (1, 2)
    assert torch.isfinite(output.logits).all()


def test_literal_cls_source_text_is_content_between_inserted_special_units():
    text = "[CLS]"
    # The content token intentionally has the same ID as inserted CLS, so the
    # source offset and unit-kind metadata must preserve the literal text bytes.
    alignment = make_alignment(text, [(0, len(text))], ids=[101])
    batch = collate_alignment_results([alignment])

    assert batch.input_ids.tolist() == [[101, 101, 102]]
    assert batch.unit_type_ids.tolist() == [[UNIT_SPECIAL, UNIT_TOKEN, UNIT_SPECIAL]]
    assert batch.unit_offsets[0] == ((0, 0), (0, 5), (0, 0))
    content_index = 1
    content_bytes = bytes(
        byte_id - 1
        for byte_id in batch.byte_input_ids[0, content_index, : batch.byte_lengths[0, content_index]]
    )
    assert content_bytes == text.encode("utf-8")
    assert sum(end - start for start, end in batch.unit_offsets[0]) == len(text)

    output = tiny_model().eval()(batch)
    assert output.logits.shape == (1, 2)
    assert torch.isfinite(output.logits).all()


def test_collator_rejects_truncated_coverage_and_empty_batches():
    truncated = make_alignment(
        "a b c",
        [(0, 1), (2, 3), (4, 5)],
        max_token_length=3,
        allow_truncation=True,
    )
    with pytest.raises(ValueError, match="incomplete or truncated"):
        collate_alignment_results([truncated])
    with pytest.raises(ValueError, match="at least one alignment"):
        collate_alignment_results([])


def test_model_returns_two_class_logits_and_adds_position_embeddings_once():
    model = tiny_model().eval()
    batch = collate_alignment_results([make_alignment("good text", [(0, 4), (5, 9)])])
    position_calls = []
    handle = model.backbone.embeddings.position_embeddings.register_forward_hook(
        lambda module, inputs, output: position_calls.append(tuple(output.shape))
    )
    try:
        output = model(batch)
    finally:
        handle.remove()

    assert output.logits.shape == (1, 2)
    assert output.last_hidden_state.shape == (1, 5, 32)
    assert output.attention_mask.tolist() == [[1, 1, 1, 1, 1]]
    assert len(position_calls) == 1
    assert torch.isfinite(output.logits).all()
    labels = torch.tensor([1])
    loss = F.cross_entropy(output.logits, labels)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert not hasattr(output, "loss")


def test_return_dict_false_backbone_is_supported_and_keeps_gradients():
    model = CanonicalByteAwareDistilBert(
        tiny_backbone(return_dict=False),
        byte_embedding_dim=8,
        byte_num_filters=4,
        byte_kernel_sizes=(2, 3),
        byte_output_dim=12,
        dropout=0.0,
    ).train()
    batch = collate_alignment_results([make_alignment("good text", [(0, 4), (5, 9)])])

    output = model(batch)
    loss = F.cross_entropy(output.logits, torch.tensor([1]))
    loss.backward()

    assert model.backbone.config.return_dict is False
    assert output.logits.shape == (1, 2)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.last_hidden_state).all()
    assert model.byte_encoder.byte_embedding.weight.grad is not None
    assert model.byte_encoder.byte_embedding.weight.grad.abs().sum() > 0
    assert model.classifier.weight.grad is not None
    assert model.classifier.weight.grad.abs().sum() > 0


def test_fusion_uses_shared_byte_cnn_and_empty_tokens_and_specials_have_no_byte_term():
    torch.manual_seed(17)
    model = tiny_model().eval()
    # The first content token has a zero-length source interval; the second
    # owns the character, and the gap between tokens owns a space.
    alignment = make_alignment("a b", [(0, 0), (0, 1), (2, 3)], ids=[11, 12, 13])
    batch = collate_alignment_results([alignment])
    captured = {}

    def capture_embedding_input(module, args, kwargs):
        captured["inputs_embeds"] = args[1]

    handle = model.backbone.embeddings.register_forward_pre_hook(
        capture_embedding_input,
        with_kwargs=True,
    )
    byte_encoder_calls = []
    byte_handle = model.byte_encoder.register_forward_hook(
        lambda module, inputs, output: byte_encoder_calls.append(tuple(output.shape))
    )
    try:
        model(batch)
    finally:
        handle.remove()
        byte_handle.remove()

    base_words = model.backbone.embeddings.word_embeddings(batch.input_ids)
    unit_types = batch.unit_type_ids[0]
    embedded = captured["inputs_embeds"][0]
    zero_byte_token = int(torch.nonzero((unit_types == UNIT_TOKEN) & (batch.byte_lengths[0] == 0))[0])
    specials = torch.nonzero(unit_types == UNIT_SPECIAL).flatten().tolist()
    assert torch.equal(embedded[zero_byte_token], base_words[0, zero_byte_token])
    assert all(torch.equal(embedded[index], base_words[0, index]) for index in specials)

    features = model.byte_encoder(
        batch.byte_input_ids,
        batch.byte_lengths,
        batch.byte_attention_mask,
    )
    normal_token = int(torch.nonzero((unit_types == UNIT_TOKEN) & (batch.byte_lengths[0] > 0))[0])
    expected_token = base_words[0, normal_token] + torch.sigmoid(model.fusion_gate) * model.token_byte_projection(
        features[0, normal_token]
    )
    torch.testing.assert_close(embedded[normal_token], expected_token)
    gap_index = int(torch.nonzero(unit_types == UNIT_GAP)[0])
    expected_gap = model.gap_type_embedding + model.gap_byte_projection(features[0, gap_index])
    torch.testing.assert_close(embedded[gap_index], expected_gap)
    assert byte_encoder_calls == [(1, batch.input_ids.shape[1], model.byte_encoder.output_dim)]
    assert sum(1 for module in model.modules() if module is model.byte_encoder) == 1


def test_gradients_reach_byte_cnn_gate_gap_embedding_projection_and_classifier():
    torch.manual_seed(21)
    model = tiny_model().train()
    batch = collate_alignment_results([make_alignment("a b", [(0, 1), (2, 3)])])
    labels = torch.tensor([1])
    loss = F.cross_entropy(model(batch).logits, labels)
    loss.backward()
    gate_value = torch.sigmoid(model.fusion_gate)

    parameters = [
        model.byte_encoder.byte_embedding.weight,
        model.byte_encoder.convolutions[0].weight,
        model.fusion_gate,
        model.token_byte_projection.weight,
        model.gap_type_embedding,
        model.gap_byte_projection.weight,
        model.classifier.weight,
    ]
    assert all(parameter.grad is not None for parameter in parameters)
    assert model.fusion_gate.shape == torch.Size([])
    assert model.fusion_gate.requires_grad
    assert 0 < gate_value < 1
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)
    assert all(parameter.grad.abs().sum() > 0 for parameter in parameters)


def test_batch_padding_does_not_change_short_example_logits():
    torch.manual_seed(23)
    model = tiny_model().eval()
    short = make_alignment("nice", [(0, 4)], ids=[18])
    long = make_alignment("a b c", [(0, 1), (2, 3), (4, 5)], ids=[19, 20, 21])

    alone = model(collate_alignment_results([short])).logits[0]
    batched = model(collate_alignment_results([short, long])).logits[0]

    torch.testing.assert_close(alone, batched, rtol=1e-5, atol=1e-6)


def test_expanded_sequence_overflow_is_rejected_without_truncation():
    model = tiny_model(max_positions=4)
    batch = collate_alignment_results(
        [make_alignment("a b c", [(0, 1), (2, 3), (4, 5)])]
    )

    with pytest.raises(ValueError, match="exceeds DistilBERT's position limit 4"):
        model(batch)


def test_forward_rejects_incompatible_or_non_batch_inputs():
    model = tiny_model()
    with pytest.raises(TypeError, match="AlignedModelBatch"):
        model(torch.ones((1, 2), dtype=torch.long))

    batch = collate_alignment_results([make_alignment("x", [(0, 1)])])
    incompatible = replace(batch, attention_mask=torch.zeros_like(batch.attention_mask))
    with pytest.raises(ValueError, match="attention_mask must be 1"):
        model(incompatible)

    fabricated = replace(batch, _alignment_provenance=None)
    with pytest.raises(ValueError, match="must be produced by collate_alignment_results"):
        model(fabricated)


def test_model_requires_explicitly_constructed_distilbert_backbone():
    with pytest.raises(TypeError, match="explicitly constructed DistilBertModel"):
        CanonicalByteAwareDistilBert("distilbert-base-uncased")
