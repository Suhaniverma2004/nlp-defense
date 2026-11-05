# Milestone 2.2 — Canonical byte-aware DistilBERT model

## Scope and API

`src/canonical_model.py` implements the canonical SST-2 model over the
Milestone 2.1 preprocessing contract. Callers tokenize the exact original
string with the selected fast tokenizer, call `align_text()`, then pass one or
more `AlignmentResult` objects to `collate_alignment_results()`. The resulting
`AlignedModelBatch` is the only input accepted by
`CanonicalByteAwareDistilBert.forward()`.

The model constructor requires an already-created `DistilBertModel`; it does
not load pretrained weights or tokenizer assets. This keeps model selection,
tokenizer identity, revisions, and offline-loading policy explicit. No
authoritative production tokenizer revision has been selected yet.

```python
from src.alignment import align_text
from src.canonical_model import CanonicalByteAwareDistilBert, collate_alignment_results

alignment = align_text(original_text, fast_tokenizer)
batch = collate_alignment_results([alignment], pad_token_id=fast_tokenizer.pad_token_id)
output = model(batch)
logits = output.logits  # [batch_size, 2]; compute cross entropy in the caller
```

## Tensor flow

Let `B` be batch size, `U` the padded expanded-unit count, `L` the longest
UTF-8 byte span in the batch, `H` DistilBERT's hidden size, and `D` the byte
encoder output dimension.

| Stage | Shape | Meaning |
|---|---|---|
| Aligned IDs and unit kinds | `[B, U]` | WordPiece IDs for content/special units; padding ID placeholder for gaps/padding; token, gap, special, and padding kinds. |
| Byte IDs, lengths, mask | `[B, U, L]`, `[B, U]`, `[B, U, L]` | Byte IDs 1–256, padding ID 0, actual lengths, and right-padding mask. |
| Shared byte CNN | `[B, U, D]` | Masked CNN representation of every token and gap byte span. |
| Word embeddings | `[B, U, H]` | DistilBERT word lookup for content and structural special token IDs. |
| Fused expanded embeddings | `[B, U, H]` | Gated byte residual for content tokens; gap-type embedding plus projected gap bytes; unchanged word embeddings for specials; zero padding vectors. |
| DistilBERT hidden states | `[B, U, H]` | Transformer output with attention over source units and masked batch padding. |
| SST-2 logits | `[B, 2]` | Two-class sentiment logits; loss is not computed inside the model. |

The collator rejects truncated or incomplete source coverage and never applies
truncation. The model checks the padded expanded sequence length against
`config.max_position_embeddings`; overflow raises an actionable `ValueError`.
Padding is added only after each alignment's source-order units and has a zero
attention mask.

## Byte encoder and fusion

`src/byte_encoder.py` contains one configurable `ByteCNNEncoder` shared by
token-owned and gap-owned spans. It uses the Milestone 2.1 mapping (byte value
plus one, ID zero reserved for padding), validates lengths and masks, masks
convolution windows that do not touch real bytes, and returns exact zero for
empty spans even when convolution/projection biases are nonzero. An optional
`max_span_bytes` is a strict cap; the encoder never truncates.

For content tokens the byte projection is zeroed for empty spans, then added
to the DistilBERT word embedding through the learned sigmoid fusion gate. The
gate is one trainable scalar whose sigmoid is shared across all hidden
dimensions and all content-token positions; it controls the magnitude of the
projected byte residual. A structural special has only its normal word
embedding. Each gap remains its own ordered unit and combines a learned
gap-type embedding with a separate projection of the shared byte-CNN feature.
The model calls the shared CNN once over the complete padded unit tensor.

The forward pass performs a word-embedding lookup, assembles the expanded
representations, then calls `DistilBertModel` with `input_ids=None` and
`inputs_embeds` set. On the verified Transformers 5.2.0 implementation,
`DistilBertModel.forward()` calls its `Embeddings` module once; that module adds
position embeddings, LayerNorm, and dropout before the transformer. Tests hook
the position-embedding layer and assert one invocation per forward pass.

Training labels and `torch.nn.functional.cross_entropy(output.logits, labels)`
remain caller-owned. This model has no attack-detection output and no
consistency loss.

## Legacy model compatibility

`src/phase5_model.py` and `src/distilbert_char_model.py` remain unchanged and
retain their existing import paths and character-ID interfaces. They are
legacy character-vocabulary architectures, not aliases for the canonical raw
UTF-8 model. New code should use `src.canonical_model`; a caller moving from a
legacy interface must replace its per-token character IDs with an
`AlignmentResult`-derived `AlignedModelBatch`. Legacy checkpoint weights are
not compatible with the canonical model.

## Verification boundaries

Architecture tests construct a tiny local `DistilBertModel` from a
`DistilBertConfig`; they do not download model weights. Existing Milestone 2.1
alignment tests remain independent. These unit tests establish tensor
contracts, masking, gradient paths, and the single positional-embedding path;
they do not establish pretrained checkpoint compatibility, tokenizer revision
selection, training quality, or production readiness. No training,
checkpoint serialization/migration, dataset changes, or attack detection is
included in this milestone.
