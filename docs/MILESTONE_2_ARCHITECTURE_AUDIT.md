# Milestone 2 — Architecture Audit and Canonical Design Proposal

**Scope:** investigation and design only. This document records the current `main` implementation and proposes a future canonical pipeline. No source code, dependencies, model weights, or datasets were changed or trained for this audit.

## 1. Executive findings

The repository has two independently implemented character-aware DistilBERT models, but SST-2 training uses only `CharAwareClassifier` → `CharAwareModel` from `src/phase5_model.py`. `src/distilbert_char_model.py` is a second, incompatible implementation used by `src/phase5_evaluation.py` and documentation examples. Both fuse a character representation into DistilBERT's embedding input in a way that adds positional embeddings twice.

The current character pathway is **not a raw-text Unicode pathway**. `prepare_inputs()` optionally sanitizes text, then tokenizes it and builds character IDs from reconstructed WordPiece token strings. SST-2 training duplicates the WordPiece-to-character logic but does not apply the inference sanitizer. Neither path carries source offsets, so token index alignment is maintained while token-to-original-character alignment is not.

`src/char_vocab.py` has an ASCII-only vocabulary. Non-ASCII code points map to `<UNK>`; the current model therefore cannot distinguish most Unicode homoglyphs, combining marks, emoji components, or format characters in its character channel. The uncased DistilBERT tokenizer also normalizes text in the token pathway.

The training script saves a complete `CharAwareClassifier.state_dict()` when validation accuracy improves, including its DistilBERT, character CNN, fusion, and classifier weights. However, no Python inference code calls `torch.load()` or `load_state_dict()`. The existing inference/smoke paths initialize fresh models, and the standalone evaluation constructs a model with randomly initialized character/fusion weights. Checkpoint metadata and an enforced compatibility loader are absent.

**Recommendation:** adopt one hybrid DistilBERT plus raw-span UTF-8-byte CNN architecture. Use a fast tokenizer's offset mapping to associate original text spans with token positions; keep uncovered source gaps as separate byte spans and explicit sequence units rather than assigning them to neighboring tokens. Retain exact input text for the byte route; never run sanitization implicitly. Fuse byte features with DistilBERT word embeddings before the model's embedding module adds positional embeddings exactly once. Keep existing import paths as compatibility wrappers during migration, then retire duplicates only after tests and checkpoint migration are complete.

## 2. Current implementation and end-to-end data flow

### 2.1 SST-2 training path

1. `experiment/prepare_sst2.py` reads `data/sst2/sst2_train.csv` and `data/sst2/sst2_validation.csv` at module import (lines 25–26, 118–120). It creates train/validation datasets and loaders at import and immediately iterates one training batch (lines 128–172). `experiment/clean_sst2.py` is a separate cleaning script: its outputs go under `data/processed/` (lines 7–9, 56–64), while the training loader reads from `data/sst2/`; the cleaning output is not wired into this loader.
2. For each dataset row, `SST2Dataset.__getitem__()` passes the sentence directly to `get_tokenizer()` and requests truncation and max-length padding to 128 tokens (lines 75–99). No call to `process_text()` or `sanitize()` is made in this path.
3. `build_char_ids()` converts token IDs back to token strings, removes the WordPiece `##` marker, iterates the resulting Python string by code point, maps at most 20 units through `get_char_idx()`, and zero-pads (lines 37–58). `[CLS]`, `[SEP]`, and `[PAD]` receive all-zero character rows.
4. `experiment/train_charaware.py` constructs `CharAwareClassifier()` (line 56), which constructs `CharAwareModel()` (see `src/charaware_classifier.py:17–24`). It trains the entire model with cross-entropy on sentiment labels and AdamW (lines 60–65, 116–147). The script executes data loading and the training loop at import time; there is no `if __name__ == "__main__"` guard.
5. On a strict validation-accuracy improvement, the script saves `model.state_dict()` to `models/charaware_sst2/charaware_sst2.pt` (lines 173–186). This state dict includes the trained character encoder and fusion parameters, but no optimizer state, epoch/config/tokenizer/vocabulary metadata, or data provenance.

### 2.2 Current `phase5_model` inference path

`prepare_inputs()` in `src/phase5_model.py:87–127` applies `process_text()` by default (`apply_defense=True`, lines 91–95), then tokenizes and pads/truncates to `max_length` (lines 97–107). It converts the resulting token IDs back into token strings and creates character IDs from those strings (lines 108–123). The default `CharAwareModel.forward()` in lines 50–70 fuses token and character embeddings and sends them to DistilBERT. `CharAwareClassifier.forward()` mean-pools the hidden states with the attention mask and applies a two-class linear head (`src/charaware_classifier.py:26–48`).

`experiment/test_classifier.py:9–35` is a smoke-test helper. If no model is supplied, it creates a fresh `CharAwareClassifier()` at line 17; it does not load the SST-2 checkpoint. `src/chatbot_security.py` now requires callers to supply a model and checkpoint identifier, but does not load or validate that checkpoint. `src/phase5_evaluation.py:18–21` creates `DistilBertCharModel` from pretrained DistilBERT weights and fresh character/fusion initialization; it does not load the SST-2 checkpoint.

### 2.3 Separate `DistilBertCharModel` path

`src/distilbert_char_model.py:21–229` defines a second model with a separate `CharEncoder` implementation, different parameter names, different fusion module, and its own forward signature. `create_char_aware_model()` at lines 224–261 builds a fresh model and loads base DistilBERT weights. It is not the model class used by `experiment/train_charaware.py`. `src/phase5_evaluation.py` uses it without loading trained character/fusion weights and writes a CSV as a module-level side effect (lines 100–117).

The architecture tests for this class construct local DistilBERT configurations and mock `from_pretrained` (`tests/test_distilbert_char.py:20–64`), which is useful for offline shape checks. The tests do not establish tokenizer/source-span correctness, Unicode preservation, or checkpoint parity.

## 3. Findings for the required architecture questions

### 3.1 Character source and Unicode preservation

- **Character source:** both active preprocessing implementations derive character IDs from `convert_ids_to_tokens()` output, not from the original raw string: `src/phase5_model.py:108–123` and `experiment/prepare_sst2.py:37–58, 93–99`. The separate model accepts caller-provided character IDs and has no raw-text encoder contract.
- **Preprocessing loss:** `src/phase5_model.py:74–84` applies `sanitize()` by default before tokenization. `src/sanitization.py:28–30` applies NFKC and a small Cyrillic homoglyph map; lines 32–36 remove some emoji and normalize whitespace; lines 47–52 compose those operations and strip edges. These operations can alter or discard distinctions before the character path runs. The training dataset does not use this sanitizer, so train/inference preprocessing differs.
- **Tokenizer loss:** even with `apply_defense=False`, `DistilBertTokenizer.from_pretrained("distilbert-base-uncased")` is used (`src/phase5_model.py:15–17, 96–106`). The uncased tokenizer lowercases and normalizes token input; character IDs are then based on its canonical WordPiece tokens. There is no preserved raw source string or offset map in the current batch contract.
- **Unknown characters:** `src/char_vocab.py:12–49` builds 100 entries from PAD/UNK, ASCII letters, digits, ASCII punctuation, and four whitespace characters. `get_char_idx()` lowercases then returns `<UNK>` (ID 1) for unmapped values (`src/char_vocab.py:52–68`). Uppercase entries are therefore not used distinctly. Non-ASCII scripts, combining marks, emoji code points, variation selectors, joiners, and zero-width format characters all collapse to the same unknown ID when they reach this lookup.
- **Special/unknown token issue:** the current builders special-case `[CLS]`, `[SEP]`, and `[PAD]`, but not `[UNK]`. If the tokenizer represents an unsupported source sequence as `[UNK]`, the character route encodes the literal ASCII spelling of that special token (lowercased by `get_char_idx()`), not the original characters.
- **Unicode sequence units:** `list(token)` iterates Python Unicode code points, not grapheme clusters or UTF-8 bytes. A decomposed base-plus-combining-mark sequence is separate code points; emoji ZWJ families and emoji plus variation selector are multiple code points. The current vocabulary collapses the unsupported elements to UNK. There is no explicit grapheme segmentation or preservation policy.
- **Sanitizer coverage:** the sanitizer does not explicitly remove ZWJ or variation selectors. Its emoji regex covers selected code-point ranges, not full grapheme sequences; associated format/variation code points can survive that regex and later be handled by tokenizer normalization. This is not a reliable raw-Unicode encoding contract.

### 3.2 Alignment, padding, truncation, and position embeddings

- **Tensor alignment:** character rows are generated from the token strings returned for the padded/truncated token IDs, so there is one character row per token position and the shapes line up. `[CLS]`, `[SEP]`, and `[PAD]` rows are all-zero in the primary preprocessing path.
- **Source alignment:** there are no `offset_mapping` values or source spans. A WordPiece such as `##ing` is reconstructed as `ing`, and its characters are not tied to an interval in the original string. Thus positional row alignment is correct, but token-to-raw-character alignment is not represented. Tokenizer normalization, dropped controls, and WordPiece segmentation make token-string reconstruction lossy.
- **Truncation:** token truncation is performed first by the tokenizer. Each reconstructed token string is separately truncated to its first 20 Python code points. This second truncation is silent and cannot preserve which original span was dropped. The training and inference limits currently agree numerically (`128` tokens and `20` char units), but live in separate implementations.
- **Position embeddings are added twice in both implementations:** `CharAwareModel.forward()` calls `self.bert.embeddings(input_ids)` at `src/phase5_model.py:64`, which already adds DistilBERT positional embeddings, LayerNorm, and dropout. It later calls `self.bert(inputs_embeds=fused)` at line 69. In the installed Transformers 5.2.0 implementation, `DistilBertModel.forward()` passes `inputs_embeds` through `Embeddings.forward()`, which adds positional embeddings and applies LayerNorm/dropout again. `DistilBertCharModel.forward()` manually adds position embeddings and applies LayerNorm/dropout (`src/distilbert_char_model.py:170–181`), then calls the same `DistilBertModel(inputs_embeds=...)` path at lines 183–186, which repeats them. This is a concrete architecture defect, not an inference from the class names.
- **Padding behavior:** the character CNNs use embedding padding index 0, but convolution biases and max pooling are not masked by actual character lengths. An all-zero character row can therefore still produce a non-zero encoder output. Token-level attention masks hide padded token positions in the transformer/pooling, but no explicit character mask is used inside either CNN.

### 3.3 Training/inference parity and checkpoint completeness

- **Preprocessing parity is broken:** SST-2 training tokenizes the raw sentence (`experiment/prepare_sst2.py:77–97`) while `prepare_inputs()` sanitizes by default (`src/phase5_model.py:87–106`). Character encoding is duplicated in these two locations rather than shared. The inference helper can disable defense preprocessing, but that is not the default and it still uses token-derived characters.
- **Checkpoint weights:** the save operation uses the complete `CharAwareClassifier.state_dict()` (`experiment/train_charaware.py:177–185`), so it does include trained DistilBERT, character CNN, fusion, and classifier tensors. However, repository source search found no `torch.load()` or `load_state_dict()` call. Consequently no checked-in inference path restores those trained values.
- **Local artifact observation:** an ignored `models/charaware_sst2/` directory is present in this workspace (`.gitignore:142` ignores `models/`). It has a PyTorch-serialization-layout `data.pkl`, `version`, and `data/` storage files and state keys including `encoder.bert.*`, `encoder.char_encoder.*`, `encoder.fusion.*`, and `classifier.*`. It is a directory rather than the `.pt` file path written by the training script; its provenance/configuration cannot be verified from repository metadata. It was inspected without loading or modifying it and should not be treated as a validated deployable checkpoint yet.
- **Metadata/compatibility:** only a state dict is saved. There is no format version, model class/fusion config, hidden size, tokenizer name/revision, tokenizer files, character vocabulary mapping/version, text preprocessing policy, character length limit, label map, training seed, dataset/split hash, or validation record bound to the checkpoint. No compatibility validation or actionable load error path exists.
- **Import and run side effects:** `experiment/train_charaware.py` runs training at import; `experiment/prepare_sst2.py` reads data and consumes a batch at import; `src/phase5_evaluation.py` loads models, runs examples, and writes results at import. This makes tests and library reuse liable to unintended I/O, downloads, and long-running computation.

## 4. Candidate architecture comparison

### 4.1 Unicode representation choices

| Representation | Unicode coverage | Alignment with DistilBERT | Compute and research fit | Complexity / trade-off |
|---|---|---|---|---|
| Raw Unicode code points | Can represent every Unicode scalar value directly; a dense table over the full range is large, while a learned finite vocabulary reintroduces OOV/UNK behavior. | Straightforward to slice original text by tokenizer offsets, but codepoint-to-token segmentation still needs offsets. | Moderate for short token spans; useful for studying specific scripts and code-point perturbations. | Requires a deliberate full-range, hashing, or fallback policy. Not equivalent to grapheme units; combining/emoji sequences remain multi-unit. |
| Grapheme clusters | Captures user-perceived units such as base-plus-mark and many emoji sequences when a Unicode grapheme segmenter is used. | More complex: a grapheme may cross tokenizer boundaries; offsets must preserve cluster boundaries and tokenizer normalization effects. | Appropriate when perturbations target visible characters or emoji sequences. | Segmentation depends on Unicode data/version and a segmenter; variable-length cluster vocabulary can still produce OOVs. |
| UTF-8 bytes | Every valid Unicode scalar sequence maps losslessly to bytes; fixed alphabet of 256 values, with a separate PAD ID. | Can encode the exact original substring selected by each token's offsets. Must convert character offsets to source substrings before UTF-8 encoding, not treat byte offsets as tokenizer character offsets. | Small finite embedding alphabet; 2–4 bytes per Unicode scalar and short CNNs are practical. Good for broad, deterministic perturbation coverage. | More sequence positions than code points and byte patterns are less linguistically interpretable. Invalid Python surrogate code points need an explicit reject/encode policy. |

### 4.2 Character sequence encoder choices

| Encoder | Strengths | Limitations / cost | DistilBERT fit |
|---|---|---|---|
| Character-level CNN | Efficient local pattern extraction; parallel over positions; existing repository experience. | Pooling can discard order/location; padding and mask handling must be correct; current variants use different convolution designs. | Strong practical fit for per-token byte spans and low incremental inference cost. |
| Character-level BiLSTM | Preserves order and contextualizes the full character/byte sequence. | Sequential compute and packing/masking complexity; slower for batches and long spans. | Compatible through a projection to DistilBERT hidden size, but less attractive for latency. |
| Character-level Transformer | Flexible long-range interactions and attention diagnostics. | Quadratic cost in per-token character length, more parameters and tuning; likely excessive for short token spans. | Technically compatible after projection, but raises compute and implementation burden. |

### 4.3 Hybrid token-character approach

A hybrid keeps pretrained WordPiece token representations as the primary linguistic channel and adds a second representation derived from the corresponding **original source spans**. It offers a controlled comparison between unchanged DistilBERT tokenization and an explicit character/byte channel. Its key complexity is alignment; offsets and tests are therefore part of the model contract, not optional preprocessing details.

## 5. Recommended canonical architecture

**Proposal:** `DistilBertRawByteCNNClassifier` (working name), a hybrid token-plus-raw-span-byte model. The byte CNN is the first canonical character encoder; other character encoders remain ablations rather than competing production implementations.

### 5.1 Canonical pipeline

1. **Input contract:** accept a batch of Python `str` values plus optional labels/sample IDs. Reject non-string values and define an explicit policy for unpaired surrogate code points; recommended default is a clear validation error. Keep the exact input string available through all stages.
2. **Unicode policy:** do not call `sanitize()`, NFKC, homoglyph mapping, emoji removal, whitespace collapsing, trimming, or case folding before the byte channel. DistilBERT's own tokenizer normalizer remains part of the token channel and must be versioned/documented. Any defense or normalization transform is an explicit experimental input transform, not hidden preprocessing.
3. **Tokenization and alignment:** use a fast tokenizer matching the pinned DistilBERT checkpoint and request `offset_mapping` together with IDs, attention masks, and structural special-token masks. Validate offsets against each original input. Do not reconstruct characters from `convert_ids_to_tokens()`. Use each content token's source interval and retain inter-token/leading/trailing source gaps as separate spans and explicit gap units so whitespace and format characters are not silently omitted or assigned to a neighboring token. Structural special tokens and padding use an empty span and zero byte contribution. Reject invalid/overlapping offsets with a diagnostic instead of guessing.
4. **Byte representation:** for each aligned source substring, UTF-8 encode it and map bytes `b` to IDs `b + 1` in `[1,256]`; reserve ID `0` for byte padding (embedding size 257). Preserve exact byte order. This representation covers homoglyphs, combining marks, emoji, variation selectors, ZWJ and zero-width code points without an open-ended vocabulary.
5. **Padding/truncation:** truncate the token sequence with the tokenizer's explicit max-token policy. Pad token/attention dimensions together. Pad byte spans to a configurable cap, but never silently crop: strict mode should raise with sample/token location; an explicitly enabled truncation mode must report overflow counts and policy. Supply byte lengths/masks so padded bytes cannot affect pooling. Empty spans map to a zero vector after the encoder; padding-only token rows must not acquire a convolution-bias representation.
6. **Character encoder:** use a modest byte embedding followed by a 1-D CNN over each token's byte sequence, masked pooling, and a projection to `config.dim`. Keep the encoder parallel over tokens and batch items. Use a small initialization scale for its projection so the pretrained token route remains stable at the start of fine-tuning.
7. **Fusion and position embeddings:** obtain only DistilBERT word embeddings (`word_embeddings(input_ids)`), fuse with projected byte features at the token positions, and pass the result to `DistilBertModel(inputs_embeds=..., attention_mask=...)` exactly once. Let DistilBERT's embeddings module add positional embeddings, LayerNorm, and dropout once. Use a simple gated residual (`word_emb + gate * byte_proj`) with a documented initialized gate; keep special/padded positions' byte contribution zero. The gate/fusion parameters are learned and checkpointed.
8. **Task objective:** for SST-2, retain two-class cross-entropy with the existing sentiment labels. If testing transformation robustness, apply only transformations whose sentiment label is verified/preserved, train/evaluate those examples explicitly, and report clean and per-transformation task metrics. A consistency loss (for example KL between clean and transformed predictive distributions) is an ablation, not assumed by default. An attack detector is a separate task requiring explicit attack labels and a separate classification head/evaluation protocol.
9. **Inference contract:** one shared deterministic encoder function must produce token IDs, attention mask, byte IDs, and byte lengths/masks for both training and inference. The inference loader returns a model in `eval()` mode and a tokenizer from the exact checkpoint bundle/revision. No implicit random initialization or fallback to a base-only or similarity model is allowed when checkpoint loading fails.

### 5.2 Checkpoint format and validation

Use a versioned checkpoint directory, not an unstructured state dict:

```text
checkpoint/
  config.json          # schema/model/task/tokenizer/Unicode/encoding settings
  model.safetensors    # complete trained model state, including byte CNN + fusion + head
  tokenizer/           # tokenizer assets/config saved at the pinned revision
  training_state.pt    # optional optimizer/scheduler/RNG state for resumption
```

`config.json` should include at least: `format_version`, canonical model class, complete DistilBERT config and model ID/revision, fast-tokenizer ID/revision, tokenizer normalizer identity, raw-input policy, offset/alignment algorithm version, byte encoding version and PAD mapping, encoder/fusion config, token/byte limits and overflow policy, task name, label map, seed, training data/split hashes, and selected validation metrics. Save all trainable weights in one model artifact. The loader must verify schema version, expected model class, tensor keys/shapes, tokenizer identity, vocabulary/encoding contract, and label-map compatibility before inference; it must fail clearly on missing or mismatched assets. Include an integrity hash/manifest. Package tokenizer assets so offline inference is possible; no silent network fetch or random-weight fallback.

For resumable training, save optimizer/scheduler, epoch/step, and RNG state separately and label it as trusted training state. Deployment inference should load only the model weights and metadata with a non-pickle format where practical.

## 6. Proposed module boundaries

| Proposed module/path | Responsibility |
|---|---|
| `src/unicode_policy.py` | Validate raw strings; document/version tokenizer-side normalization; explicit opt-in transformations only. |
| `src/alignment.py` | Fast-tokenizer calls, offset validation, source-gap assignment, special/pad handling, deterministic truncation reports. |
| `src/byte_encoding.py` | UTF-8 byte IDs, lengths/masks, padding/cap policy, versioned encoding metadata. |
| `src/canonical_model.py` | Single canonical DistilBERT + byte CNN + fusion + task head; hidden size from DistilBERT config. |
| `src/checkpoint.py` | Versioned save/load, manifest/integrity checks, strict compatibility diagnostics, offline tokenizer restore. |
| `experiment/prepare_sst2.py` | Side-effect-free dataset and DataLoader factories using the shared encoder contract. |
| `experiment/train_charaware.py` | Guarded CLI/function-based training, reproducible configuration, task evaluation, checkpoint save/resume. |
| `experiment/infer_charaware.py` | Checkpoint-required inference entry point using the same tokenizer/encoding code as training. |
| `tests/test_unicode_encoding.py` | Unicode, alignment, padding/truncation, deterministic offline fixtures. |
| `tests/test_checkpoint_roundtrip.py` | Offline local-config save/load and inference parity. |

Keep `src/phase5_model.py`, `src/distilbert_char_model.py`, and `src/charaware_classifier.py` as thin deprecated compatibility facades during migration. Do not delete them until callers, docs, checkpoints, and import contracts have been migrated and tested.

## 7. Training and inference contracts

### Training

- Input rows are immutable raw text plus task labels and stable IDs. Preserve the raw strings in dataset outputs and record exact transformations/labels when augmentation is used.
- Training and validation use the same tokenizer revision, source-offset algorithm, byte encoding version, token/byte budgets, and empty-span handling.
- Training has no import-time data loading or optimization. Seed Python/PyTorch/DataLoader workers, record configuration, and use validation only for model selection. Keep a held-out test split for final reporting.
- Main task objective is per-example cross-entropy for SST-2. Report clean and transformation-stratified metrics; do not infer detector quality from embedding similarity.

### Inference

- `load_checkpoint(path, device, local_files_only=True)` is required; it verifies metadata and loads all model weights strictly. Missing checkpoint, tokenizer files, incompatible architecture, encoding version, or label map is a hard actionable error.
- `encode_batch()` is shared with training and uses exact input text for source-span bytes. Inference returns logits/probabilities and model/checkpoint provenance, not claims unsupported by the trained task.
- Deterministic evaluation uses `model.eval()` and inference/no-grad mode. Truncation/offset warnings are returned or raised, never hidden.

## 8. Test strategy (offline and deterministic)

Use a local fast tokenizer fixture with explicit vocabulary/offset behavior or a deterministic fake tokenizer; use a tiny locally initialized `DistilBertConfig`. Disable network access in unit tests and assert no `from_pretrained()` call is needed. Fix Python/PyTorch seeds where randomly initialized modules are compared. Expected IDs/spans should be specified independently in fixtures.

| Case | Required assertion |
|---|---|
| ASCII and empty string | ASCII bytes round-trip exactly; empty input yields valid special/pad tensors and zero byte features. |
| Unicode homoglyph and mixed scripts | Raw source bytes differ and remain distinguishable even if token IDs normalize to the same sequence. |
| Combining marks | Composed and decomposed strings are not silently merged in the byte channel; offsets cover their source spans. |
| Zero-width characters | U+200B/U+200D survive raw byte encoding and are assigned to a token/span or explicitly reported as unaligned. |
| Emoji and variation selectors | Emoji and VS bytes are represented; emoji selectors are not dropped by grapheme or byte boundary logic. |
| ZWJ sequences | Entire multi-code-point sequence is covered exactly once across aligned spans; no code point is lost or duplicated. |
| Token alignment | Every content token has valid offsets; expected source substring bytes match fixture; special tokens have empty spans. |
| Whitespace/leading/trailing gaps | Source partition includes explicitly assigned whitespace, with no overlap or omission. |
| Padding | PAD byte ID is 0; masks/lengths exclude padded bytes; all-padding/special rows produce zero byte contribution. |
| Long input/token | Token truncation and byte overflow are deterministic, surfaced, and match the configured policy. |
| Training/inference parity | Shared encoder returns bit-identical IDs/masks for the same text/config in train and inference entry points. |
| Position embeddings | Instrumented/local model test proves positional embeddings and embedding LayerNorm are applied exactly once. |
| Checkpoint round trip | Save local config, reload offline, verify all keys/config and logits match within deterministic tolerance. |
| Checkpoint incompatibility | Wrong format, vocabulary, tokenizer revision, or missing tensor raises a specific error rather than partial loading. |
| Compatibility facades | Existing public class/function imports and supported signatures map to the canonical model without silent behavior changes. |

Existing tests cover local model shapes/gradients and basic inference contracts (`tests/test_distilbert_char.py`, `tests/test_char_encoder.py`, `tests/test_inference_contracts.py`), plus dataset cleaning (`tests/test_clean_sst2.py`). They do not cover the Unicode/alignment/checkpoint cases above. `tests/test_sanitization.py` tests sanitizer behavior, not preservation by the character model.

## 9. Incremental implementation plan

1. **Contract and fixture stage:** approve raw-text/normalizer policy, offset ownership (including whitespace), byte overflow behavior, task objective, and checkpoint schema. Add deterministic expected-span fixtures before implementing the encoder.
2. **Shared preprocessing stage:** implement `unicode_policy.py`, `alignment.py`, and `byte_encoding.py`; test all Unicode sequences, offsets, empty inputs, padding, and truncation offline. Keep legacy encoders untouched.
3. **Canonical model stage:** implement `canonical_model.py` with local-config support, one positional embedding path, masked byte CNN pooling, and gated residual fusion. Add unit tests for shapes, masks, gradients, and single position addition.
4. **Dataset/training stage:** refactor `experiment/prepare_sst2.py` into side-effect-free factories, route both splits through the shared encoder, explicitly select the intended cleaned/raw CSV paths, and guard training behind a CLI/main entry point. Establish seeds and task metrics.
5. **Checkpoint/inference stage:** add versioned save/load and metadata validation; prove offline round-trip equality. Replace inference defaults with checkpoint-required entry points and clear errors.
6. **Evaluation/ablation stage:** evaluate clean and label-preserving transformed test examples; compare baseline DistilBERT, byte CNN, and selected codepoint/grapheme alternatives with fixed data/splits and report transformation-stratified task metrics. Keep similarity diagnostics separate.
7. **Compatibility/migration stage:** migrate docs and callers, add facade regression tests, inspect existing local checkpoint provenance/compatibility, and only then decide whether old modules can be retired. No checkpoint should be silently converted or loaded under a different architecture.

## 10. Risks and unresolved research decisions

- **Tokenizer offsets under normalization:** fast DistilBERT offset mappings must be empirically validated for accent stripping, Unicode normalization, unknown tokens, and controls; tests should be pinned to the exact tokenizer revision.
- **Whitespace ownership:** tokenizers do not usually emit whitespace tokens. The proposed gap-to-token rule must be chosen, documented, and ablated because attaching whitespace to a neighboring token affects the character representation.
- **Byte vs grapheme inductive bias:** bytes guarantee coverage but may need more data to learn visual/semantic character families. Compare byte and grapheme/codepoint variants under identical splits before claiming superiority.
- **Byte/sequence budgets:** byte spans can be up to four bytes per Unicode scalar and may be long for unknown tokens. Caps affect worst-case coverage and must be surfaced in metrics.
- **Normalization channel effects:** preserving raw bytes while using an uncased, normalizing token branch creates intentional dual views. Whether to use a cased tokenizer is a research ablation, not a hidden setting.
- **Label preservation:** some generated transformations may change meaning or sentiment. Augmentation requires explicit review/labels; security attack detection is a separate supervised task.
- **Checkpoint provenance:** the ignored local checkpoint artifact has no repository-tracked provenance and its path differs from the training script's expected `.pt` path. Verify it independently before any migration or evaluation.
- **Compute/reproducibility:** byte CNNs add memory/latency; seeds alone do not guarantee bitwise determinism on all accelerators. Record runtime/hardware and deterministic settings.
- **Compatibility:** the two current models have different parameter names/shapes. Existing state dicts are not directly interchangeable; migration must be explicit and validated.

## 11. Audit completion and boundary

The original architecture audit was design-only. Milestone 2.1 subsequently implemented the shared Unicode validation, byte encoding, and tokenizer/source-alignment utilities documented below. This milestone did not modify existing model classes, dependencies, training code, checkpoints, or datasets, and did not train a model or download tokenizer assets.

## Architecture Decisions and Implementation Contracts

This section records the design contracts for the proposed first model implementation. Milestone 2.1 implements and tests the Unicode validation, byte encoding, and alignment preprocessing contracts; the byte CNN, model fusion, training, checkpoint, and inference contracts remain design requirements. Real-tokenizer checks currently characterize only the cached revision documented below.

### A. Terminology and scope

- A **Unicode code point** is one Unicode scalar value in a string. A single visible symbol may contain multiple code points (for example, a base letter followed by a combining mark).
- A **grapheme cluster** is a user-perceived text unit defined by Unicode grapheme-boundary rules. It can contain multiple code points, as with many emoji ZWJ sequences and accented letters represented with combining marks.
- A **UTF-8 byte** is an 8-bit unit in the encoding of a string. It is not a Unicode character: each valid Unicode scalar value encodes to one to four UTF-8 bytes, and a grapheme cluster can encode to many bytes.

The first implementation uses **only UTF-8 bytes** for the raw-text pathway. Code-point and grapheme encoders are future research ablations; they are not additional first-stage implementations or selectable production modes.

### B. Input and Unicode transformation contract

1. The input to the shared encoder is the exact original Python `str`. It is passed unchanged to the tokenizer and retained unchanged as the source for byte spans. The encoder does not call `sanitize()`, NFKC/NFC, case folding, whitespace normalization, emoji removal, homoglyph mapping, trimming, or control-character filtering.
2. The byte branch takes `text[start:end]` slices from that exact string and encodes each slice with strict UTF-8. It performs no Unicode normalization or replacement. A Python string containing an unpaired surrogate is rejected with the offending sample/character location; it is not silently replaced or encoded with a lossy error handler.
3. The DistilBERT token branch applies only the normalization and segmentation that belong to the saved tokenizer configuration. For the standard intended `distilbert-base-uncased` fast-tokenizer setup, the expected stages are BasicTokenizer-style text cleaning, lowercasing/accent handling and Chinese-character boundaries, whitespace/punctuation pre-tokenization, then WordPiece segmentation. The exact serialized normalizer/pre-tokenizer configuration and tokenizer revision are authoritative and must be recorded and validated; no separate application-level normalization runs before tokenization. Do not assume NFKC or the repository's homoglyph map is applied by the tokenizer.
4. Experimental transformations are explicit upstream operations. Each transformed string is treated as a new exact raw input and is paired with its original sample ID, transform name/version, and label-preservation annotation. The original string is retained for audit. Applying a transform never overwrites the clean source.

### C. Exact token-to-byte alignment contract

The shared preprocessing output contains token IDs, attention mask, structural-special mask, token offsets, token-byte spans, gap-byte spans, ordered model units, and a coverage report. Offsets use **half-open Python-string code-point indices** `[start, end)` into the exact original string supplied to that tokenizer call.

#### Offset validation

- Require a fast tokenizer and request offsets and structural special-token metadata in the same call that tokenizes the unchanged source string. `offset_mapping`, `input_ids`, and the special-token mask must have identical sequence lengths.
- Every offset endpoint must be an integer (not a boolean) and satisfy `0 <= start <= end <= len(original_text)`. Validate the result against Python `len()` and `original_text[start:end]`; do not interpret the offsets as UTF-8 byte positions and do not compare the source slice to a decoded WordPiece string, because tokenizer normalization is expected to change the token surface.
- The tokenizer call must receive the exact original string, not a normalized copy. The source string associated with offsets is immutable throughout encoding; an optional source digest may be recorded for debugging/reproducibility.
- Structural special tokens inserted by the tokenizer postprocessor and padding must have empty source offsets and contribute no bytes. Use the postprocessor's structural-special mask together with offsets; do not classify a token as source-free merely because its spelling or ID is `[UNK]`. An `[UNK]` with a non-empty source interval is a content token and receives the bytes from that original interval. A literal special-token spelling in user text is content if it has a non-empty source interval.
- Content-token intervals must be ordered and non-overlapping. Any out-of-range, reversed, or overlapping non-empty offsets reject that sample with the token indices and ranges in the error. Do not de-duplicate overlaps by arbitrarily choosing a token owner; such offsets make one-to-one ownership ambiguous.
- A zero-length content-token offset owns no source characters. Any source characters it appears to omit remain eligible for a gap span.

#### Source ownership and gaps

For each accepted, non-truncated example, take the union of all valid non-empty content-token intervals. Its complement in `[0, len(original_text))` is partitioned into maximal **gap spans**. This includes leading whitespace `[0, first.start)`, inter-token whitespace or omitted text `[previous.end, next.start)`, trailing whitespace `[last.end, len(text))`, and any non-whitespace source characters not covered by tokenizer offsets. Empty input has no source spans and therefore no gap unit.

Each content-token span and each gap span is UTF-8 encoded exactly once. Preserve their source order and represent each non-empty gap as a separate gap unit (with its own unit type and byte sequence) in the model sequence. Do **not** concatenate gaps into the preceding or following token's byte span. This keeps whitespace and tokenizer-omitted characters visible without falsely attributing them to a lexical token. If the tokenizer offset for a normalized token spans the original characters that produced it, those original bytes belong to that token span; if an original interval has no token owner, it belongs to a gap span.

The required coverage invariant is:

> For every accepted input, the ordered token and gap source intervals form a disjoint partition of `[0, len(original_text))`. Concatenating their raw source slices in order reproduces the exact original Python string, and concatenating their strict UTF-8 encodings reproduces `original_text.encode("utf-8")` byte-for-byte. Every original code point/byte is therefore represented exactly once in the raw-byte pathway. Any character that cannot be assigned because of malformed/ambiguous offsets, an unsupported input value, a configured byte cap, or model-sequence truncation must be recorded with its exact source range and cause, or the example must be rejected. No silent loss or duplication is allowed.

This coverage check also distinguishes expected tokenizer normalization from an alignment failure: normalization may change token IDs, but cannot change the retained original string. Source text absent from content-token spans becomes a gap span. Out-of-bounds or overlapping offsets are rejected. A gap is not silently discarded just because its contents normalize away in the token branch.

#### Special cases and truncation

| Case | Contract |
|---|---|
| Leading whitespace | Gap span before the first content-token interval; separate gap unit before that token. |
| Trailing whitespace | Gap span after the last content-token interval; separate gap unit after that token. |
| Inter-token whitespace | Complement interval between adjacent content-token spans; separate gap unit at that source boundary. |
| Tokenizer-omitted/normalized text | If included inside a token's source interval, those original bytes remain in that token's byte input. Otherwise they form an explicit gap span/unit. Report gap ranges and code-point counts by category where classifiable; do not infer the reason solely from content. |
| Overlapping offsets | Reject the sample and report token indices/ranges. Do not duplicate bytes or select an arbitrary winner. |
| Empty string | Accept as zero source characters; structural `[CLS]`/`[SEP]` units may remain, with empty byte spans and valid masks. |
| Structural special tokens/padding | Empty source interval, no byte CNN contribution. A non-empty offset always takes precedence as evidence of source text, including for `[UNK]`. |
| Unknown token | Encode its original non-empty offset span as UTF-8 bytes; never encode the literal `[UNK]` spelling in place of source text. |
| Token truncation | Tokenize without hidden truncation to obtain coverage, then enforce a documented total model-position budget after adding gap units. Strict default rejects an example if source intervals would be omitted. If an explicitly enabled truncation mode is used, report exact unrepresented source ranges and counts in `coverage_report`; the result is marked truncated and cannot be mistaken for full-coverage input. |
| Byte-span cap | Strict default rejects an over-cap unit with its sample, unit index, and source range. If explicit cropping is later approved, report exact omitted byte/code-point ranges; never silently crop. |

Separate gap units are preferred over gap-to-neighbor assignment because the latter makes the token byte span's ownership ambiguous and can make whitespace perturbations appear to belong to a lexical item. The cost is a longer model sequence: the configured maximum must apply to the expanded ordered sequence of structural specials, token units, and gap units, not just the tokenizer's original WordPiece count.

### D. Byte CNN and fusion contract

- UTF-8 byte IDs are `byte_value + 1` (`1..256`); ID `0` is PAD only. A byte-embedding table therefore has 257 rows. The preprocessing output also carries byte lengths/masks.
- The same byte CNN encodes token spans and gap spans. Masked pooling must exclude padded byte positions. Empty spans and structural special/padding rows yield exactly zero byte contribution; convolution bias must not create a non-zero representation for an empty span.
- Token content unit: `word_embedding(input_id) + sigmoid(gate) * byte_projection(byte_cnn(token_bytes))`. Structural specials use their normal DistilBERT word embedding without byte contribution. Gap unit: a learned gap-type embedding plus a separate projection of that gap's CNN vector. All units share the ordered source position in one expanded model sequence.
- Pass the assembled `inputs_embeds` and unit attention mask into `DistilBertModel` once. DistilBERT's embedding layer adds positional embeddings, LayerNorm, and dropout exactly once. The canonical model must not call the complete embedding module and then call it a second time through `inputs_embeds`.
- The expanded sequence length, including gap units and special tokens, must not exceed the saved DistilBERT positional limit. Strict mode rejects/reports overflow under the truncation rule above. The fusion gate initialization and gap-type embedding initialization are part of the versioned model config.

### E. SST-2 task and evaluation contract

The initial objective remains **two-class SST-2 cross-entropy**. No consistency loss is introduced by default.

- **Clean validation:** use the validation split for model selection and report mean cross-entropy loss, accuracy, positive-class precision/recall/F1, confusion matrix, and example count. Record class label mapping and averaging conventions.
- **Clean held-out test:** evaluate the selected checkpoint once on a test split excluded from training and model selection; report the same metrics and support counts. Do not tune thresholds or architecture against this result.
- **Transformation-stratified task metrics:** construct transformed versions of held-out examples with stable parent IDs. For every transform, report support, count of exact no-ops, coverage/truncation/rejection counts, loss, accuracy, positive-class precision/recall/F1, confusion matrix, and paired clean-to-transformed prediction changes. Report the clean subset alongside each transform; do not merge transformations into one undifferentiated score.
- **Verify label preservation:** define each transform/version in advance. Exact no-ops are recorded separately. For transforms that can affect meaning, use independent human review of clean/transformed pairs with blinded sentiment labels and adjudication for disagreements; audit all examples where feasible or a documented representative sample. Exclude non-preserving transformations from SST-2 augmentation/robustness scoring or obtain a corrected label. Store the decision, annotator/adjudicator provenance, and original parent label. Transformations are not assumed label-preserving by their name.
- **Keep attack detection separate:** SST-2 predicts sentiment, not attack presence. Any detector requires a separately labeled attack/no-attack dataset, its own trained output head/objective, independent train/validation/test splits, and detection-appropriate metrics. Neither sentiment accuracy nor cosine similarity is attack-detection performance.

### F. Checkpoint compatibility contract

The versioned checkpoint bundle described in Section 5.2 is authoritative. It must include the complete DistilBERT backbone weights, byte embedding/CNN, token and gap projections/type embeddings, fusion gate, and SST-2 classifier head, as well as tokenizer files and metadata. The metadata must specify `format_version`, canonical architecture ID, tokenizer model/revision and normalizer config, offset/alignment algorithm version, UTF-8 byte ID mapping, gap-unit policy, token/expanded-position/byte limits, overflow policy, model config, label map, and training/data provenance.

Loading is strict: resolve tokenizer and architecture from checkpoint metadata, verify schema and all required tensors/shapes, verify the preprocessing/encoding/alignment versions, and call `load_state_dict(..., strict=True)` only after those checks. Verify a manifest/hash for the checkpoint files. Do not instantiate a random character/fusion module as fallback, silently load only the pretrained DistilBERT base, or use a checkpoint with an incompatible vocabulary/fusion/tokenizer contract. Missing files, unsupported format versions, mismatched revisions/configuration, integrity failure, and missing/unexpected state keys are explicit actionable errors. Offline inference uses packaged tokenizer assets and performs no implicit download.

The ignored local checkpoint-like directory noted in Section 3.3 is not yet compatible by evidence alone: it predates this schema, its provenance is unknown, and its current model architecture is not the proposed architecture. It must be inspected and validated in a separately approved migration task before use.

## Milestone 2.1 tokenizer validation status

The locally available `distilbert-base-uncased` fast-tokenizer assets were
inspected without network access from the Hugging Face cache. The cache's
`main` reference resolved to snapshot
`12040accade4e8a0f71eabdb258fecc2e7e948be`; its `tokenizer.json`,
`tokenizer_config.json`, `vocab.txt`, and `config.json` were present. The
snapshot's serialized normalizer is `BertNormalizer(clean_text=True,
handle_chinese_chars=True, strip_accents=None, lowercase=True)`. The repository
does not otherwise pin a tokenizer commit, so this observed cache snapshot is
not yet an authoritative production tokenizer revision.

Offline characterization used `local_files_only=True` at that exact snapshot.
For decomposed `e` plus COMBINING ACUTE ACCENT, the tokenizer returned an
offset for the base letter and omitted the accent from token offsets; alignment
retained the accent as a separate gap. The emoji `🧪` became `[UNK]` with a
non-empty `(0, 1)` offset and was aligned to its original UTF-8 bytes. A literal
`[CLS]` in user text had a non-empty `(0, 5)` content offset and a non-special
mask entry, while the inserted boundary `[CLS]` and `[SEP]` positions had empty
offsets and structural-special mask entries. A zero-width space by itself had
no content token and was retained as a full-source gap. Deterministic tests
exercise these observed cases when this exact snapshot is already cached;
they skip rather than download if it is absent.

This validates the observed snapshot only. Before model training or deployment,
select and record an authoritative tokenizer revision, then rerun the same
normalization, omitted-character, unknown-token, and literal-special tests
against its exact assets. Other tokenizer revisions or normalizer settings
remain unverified.

The minimum supported Python version is **Python 3.10**, documented in
`docs/DEVELOPMENT.md`; CI runs Python 3.13 and the clean environment was
verified on CPython 3.14.2.

## SST-2 Dataset Provenance Snapshot

The local files under `data/sst2/` are consumed by `experiment/prepare_sst2.py`. The repository's `experiment/dataset_loader.py` documents their intended source as the official GLUE SST-2 dataset, loaded with `datasets.load_dataset("glue", "sst2")`; it maps label `0` to `Negative` and `1` to `Positive`, then writes train and validation CSVs. This records repository-declared provenance, not an independent upstream checksum verification. Running that loader would download data and was not part of the audit.

| Local file | Rows | Columns | Label counts (`0`, `1`) | SHA-256 |
|---|---:|---|---:|---|
| `data/sst2/sst2_train.csv` | 67,349 | `sentence`, `label`, `idx`, `label_name` | 29,780; 37,569 | `e224c70ad9231d96187b25759c2a3aecb5fb2a43bd44218d1fba9076caa8f774` |
| `data/sst2/sst2_validation.csv` | 872 | `sentence`, `label`, `idx`, `label_name` | 428; 444 | `4eb26cca066be69a8ad4a98cc9f0d1e024b60d7dc7a293393bdfe4ae1004b131` |

No labeled held-out test CSV is present in `data/sst2/`, and `experiment/dataset_loader.py` writes only train and validation splits. The official GLUE SST-2 test split does not provide labels for supervised final metrics. Do not reuse validation as test or invent test labels. Before final held-out evaluation, approve one of these choices: obtain a separate labeled test set with provenance, or reserve a reproducible portion of the available labeled training data before any model selection and record that split/hash. The second choice reduces the training data and must be explicit.
