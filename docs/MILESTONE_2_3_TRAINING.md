# Milestone 2.3 — Reproducible SST-2 data, training, and checkpoints

This milestone adds a reproducible CPU/GPU training and evaluation pipeline for
the canonical UTF-8 byte-aware DistilBERT model and a separate DistilBERT-only
baseline. It does **not** report a full training experiment. The pipeline has
been tested with tiny, randomly initialized local models and synthetic data.

## Dataset provenance and split decision

The repository's `experiment/dataset_loader.py` records the intended source as
Hugging Face `datasets.load_dataset("glue", "sst2")`. This is repository-
declared provenance: the local data has not been re-fetched or independently
matched to an upstream release. No dataset license file or license metadata is
present in this repository; check the upstream GLUE/SST-2 terms before
redistributing the data.

The inspected files are:

| File | Rows | Labels (negative 0 / positive 1) | Missing values | Exact-text duplicate rows / groups | SHA-256 |
|---|---:|---:|---:|---:|---|
| `data/sst2/sst2_train.csv` | 67,349 | 29,780 / 37,569 | 0 | 742 / 371 | `e224c70ad9231d96187b25759c2a3aecb5fb2a43bd44218d1fba9076caa8f774` |
| `data/sst2/sst2_validation.csv` | 872 | 428 / 444 | 0 | 0 / 0 | `4eb26cca066be69a8ad4a98cc9f0d1e024b60d7dc7a293393bdfe4ae1004b131` |

Both files use `sentence`, `label`, `idx`, and `label_name`. The original
train/validation files have no exact-text overlap. The original validation
file is preserved in place, is not mixed into any generated split, and is not
used by the Milestone 2.3 training command.

The training source has five exact-text groups with conflicting labels (10
rows total). All rows in those groups are written to `excluded_ambiguous.csv`
and excluded. Of the remaining duplicate groups, one source row is retained
and the other 366 same-label duplicate rows are written to
`excluded_same_label_duplicates.csv`. This avoids label ambiguity, repeated
examples across partitions, and duplicate-text leakage. No labels are changed
or fabricated.

Seed 42 produces these partitions from the deduplicated labelled training
source:

| Split | Rows | Label 0 | Label 1 | File SHA-256 |
|---|---:|---:|---:|---|
| Train | 53,577 | 23,713 | 29,864 | `7109cb760d5d712535b5bf8b880f7ec7b4d5f6c6b1ca7406224976e8186c36e8` |
| Validation | 6,698 | 2,965 | 3,733 | `f688afcc6e3095b30578f6c93228ae7343a65f71e8c02df8cba85ac537e06d49` |
| Held-out test | 6,698 | 2,965 | 3,733 | `418f4eb878ce81f27ec185ac9fed0243fb8ca82d9848e83bb0122bb25b720121` |

All pairwise exact-text overlaps are zero. The split manifest records the
source hashes, output hashes, row counts, label counts, excluded-row counts,
seed, proportions, split algorithm, and software versions at
`data/processed/milestone2.3-seed42/manifest.json`. Split outputs are generated
and ignored by Git; the code and manifest are reproducible from the source
files. The split builder refuses to overwrite a non-empty output directory.

**Historical-use limitation:** the legacy `experiment/train_charaware.py`
trainer reads the full original training CSV, and an ignored local legacy
checkpoint exists whose run history cannot be verified. The new test partition
is held out from Milestone 2.3 runs, whose trainer receives only `train.csv`
and `validation.csv`. It is not described as globally untouched across all
prior project experiments. No legacy checkpoint is loaded or modified.

## Model and tokenizer identity

The cached snapshot
`distilbert-base-uncased@12040accade4e8a0f71eabdb258fecc2e7e948be` was checked
offline. It contains the DistilBERT configuration, `model.safetensors`,
`tokenizer.json`, `tokenizer_config.json`, and `vocab.txt`. Loading with
`HF_HUB_OFFLINE=1`, `local_files_only=True`, and `use_fast=True` succeeded; the
loaded tokenizer reports `is_fast=True`, and the model has hidden size 768,
vocabulary size 30,522, and 512 position embeddings. Loading its masked-language
model weight file into `DistilBertModel` reports five unused MLM-head tensors;
the DistilBERT backbone tensors load successfully.

That immutable revision is the pipeline's explicit reproducibility default
because it is locally available and was previously used to characterize
offsets. This does **not** select or validate it as the production tokenizer
revision. The code requires 40-character commit SHAs for both model and
tokenizer revisions and never accepts a mutable `main` ref. Supply another
reviewed immutable revision explicitly before a production research run.
Asset loading is local-only by default. The CLI's `--allow-hub-download` flag
is the explicit opt-in for fetching the configured revisions when they are not
already cached.

## Data and preprocessing interfaces

- `src/sst2_data.py`: validates raw text and labels, inspects local files,
  creates stratified splits, records provenance, and verifies split hashes.
- `src/training.py`: immutable `TrainingConfig`, seeded loader construction,
  local/pretrained asset loading, the DistilBERT-only baseline, shared aligned
  batching, training, validation, and clean classification metrics.
- `src/canonical_model.py` and Milestone 2.1 modules: canonical model and the
  existing Unicode validation, tokenizer-offset alignment, UTF-8 byte, and
  model-batch contracts. Preprocessing is not duplicated.
- `src/checkpointing.py`: versioned safetensors model bundles, packaged
  tokenizer assets, integrity hashes, strict model state loading, and optional
  separately trusted training-state files.
- `src/evaluation.py`: post-selection validation/test evaluation tied to the
  exact split hashes recorded by the checkpoint; writes JSON and CSV metrics.

Both models use `sentence` and binary `label` columns (`0 = Negative`,
`1 = Positive`). `ClassificationCollator` passes each original string through
`align_text()` without normalization. Strict token/gap alignment and byte-span
limits reject overflow; the pipeline does not silently truncate. The baseline
uses the same aligned tokenization and token budget, then consumes the native
token sequence; the canonical model consumes the expanded token/gap batch.
Both return two-class logits and use external cross-entropy.

The current baseline is a standard DistilBERT CLS-pooled classifier, while the
canonical system mean-pools expanded token/gap representations and adds the raw
byte pathway. This is a **whole-system comparison**; it does not isolate the
causal effect of byte features. A byte-free ablation with matched expanded-unit
pooling and head behavior is not included in this milestone. Do not attribute a
performance difference solely to byte features.

The configuration records: model/tokenizer IDs and commit revisions, expanded
and byte-span limits, microbatch size, gradient accumulation steps, seed,
epochs, learning rate, weight decay, warmup ratio, gradient clip norm, mixed
precision, device, worker count, output directory, and canonical byte-CNN
dimensions. Python, NumPy, PyTorch CPU/CUDA, and DataLoader generator RNGs are
seeded and saved in the resume state. CUDA uses deterministic cuDNN settings
where available; bitwise equality across devices, drivers, and library versions
is not promised. Resume requires the same recorded Python/library versions and
resolved device as the saved run. Unsupported CUDA mixed precision is disabled
with a warning; CPU remains supported.

Gradient accumulation weights each microbatch loss by its example count within
the optimizer update, including a smaller final microbatch or incomplete final
accumulation group. Gradients are clipped and the scheduler advances only at
optimizer-update boundaries. The configured effective batch size is
`batch_size * gradient_accumulation_steps` except for the final partial group.
Training uses AdamW with betas `(0.9, 0.999)`, epsilon `1e-8`, configured
weight decay, and `amsgrad=False`, followed by a LambdaLR linear warmup/decay
schedule. Total scheduler steps are derived from train rows, microbatch size,
accumulation, and epochs; the comparison validator reports these effective
optimizer/scheduler settings and rejects mismatched shared controls.

Validation runs at `--validation-interval-steps` when set, and always at each
epoch end. The best checkpoint is chosen by minimum clean validation loss.
Metrics include mean loss, accuracy, positive-class precision/recall/F1, a
fixed-label `[0, 1]` confusion matrix, and per-class support. The training API
accepts no test frame. The held-out test is evaluated only through the separate
evaluation interface after checkpoint selection. Clean SST-2 classification
metrics do not establish adversarial robustness or attack detection.

## Prepare splits

From the repository root:

```powershell
python -m experiment.prepare_milestone23_splits `
  --train data/sst2/sst2_train.csv `
  --original-validation data/sst2/sst2_validation.csv `
  --output-dir data/processed/milestone2.3-seed42 `
  --seed 42
```

The command prints the source inspection and split manifest. It fails rather
than overwriting a non-empty destination. To use another seed, select a new
empty output directory.

## Train the DistilBERT baseline

```powershell
python -m experiment.train_sst2 `
  --model-kind baseline `
  --split-dir data/processed/milestone2.3-seed42 `
  --output-dir outputs/milestone2.3/baseline-seed42 `
  --model-id distilbert-base-uncased `
  --model-revision 12040accade4e8a0f71eabdb258fecc2e7e948be `
  --tokenizer-id distilbert-base-uncased `
  --tokenizer-revision 12040accade4e8a0f71eabdb258fecc2e7e948be `
  --seed 42 --epochs 3 --batch-size 8 --gradient-accumulation-steps 2 `
  --learning-rate 0.00002 --max-expanded-length 128 --max-byte-span-length 512 `
  --validation-interval-steps 1000 --mixed-precision fp16 --device auto
```

## Train the canonical byte-aware model

Use the same split directory, model/tokenizer IDs and revisions, seed, batch
and accumulation sizes, validation interval, precision, and training settings.
The commands below use the same effective batch size of 16 and the same
selection opportunities. Choose a new empty output directory:

```powershell
python -m experiment.train_sst2 `
  --model-kind canonical `
  --split-dir data/processed/milestone2.3-seed42 `
  --output-dir outputs/milestone2.3/canonical-seed42 `
  --model-id distilbert-base-uncased `
  --model-revision 12040accade4e8a0f71eabdb258fecc2e7e948be `
  --tokenizer-id distilbert-base-uncased `
  --tokenizer-revision 12040accade4e8a0f71eabdb258fecc2e7e948be `
  --seed 42 --epochs 3 --batch-size 8 --gradient-accumulation-steps 2 `
  --learning-rate 0.00002 `
  --max-expanded-length 128 --max-byte-span-length 512 `
  --validation-interval-steps 1000 --mixed-precision fp16 --device auto
```

If assets are absent from the local cache, append `--allow-hub-download` to
explicitly fetch those exact revisions. The CLI saves an experiment config,
JSON/CSV training history (updated during the run), validation-selected
immutable checkpoint directories, and `best_checkpoint.json`. At each completed
epoch it also saves an immutable
`resume-epoch-NNNN-step-NNNNNNNN-<unique>/` checkpoint containing
safetensors weights and `training_state.pt`, and updates
`checkpoints/latest_checkpoint.json`. Training state includes optimizer,
scheduler, scaler, Python/NumPy/Torch CPU/CUDA RNGs, DataLoader generator state,
epoch/global step, and best-checkpoint selection metadata. Training state uses
trusted PyTorch serialization and should only be resumed from a trusted
checkpoint directory. A fresh run refuses a non-empty output directory.

Checkpoint directories are built under unique sibling staging names. Model,
tokenizer, hashes, resume state, and the embedded best-validation checkpoint
are validated before a resume bundle is renamed to its final immutable name.
Recovery scans valid completed bundles and complete abandoned staging bundles,
ignores incomplete or corrupt ones, then replaces the latest pointer with the
newest validated epoch/step. Pointer contents are advisory; validation of the
immutable snapshot is authoritative. Incomplete staging folders are preserved
for inspection and do not block later attempts. JSON and CSV histories are
written to unique temporary files and replaced; JSON is the history commit
marker. If published JSON history is missing or corrupt on resume, the history
embedded in the validated epoch snapshot is restored. These replacements give
atomic visibility on filesystems that implement same-filesystem rename
atomically; they do not by themselves guarantee power-loss durability.

Resume is supported **at completed epoch boundaries**, not mid-epoch. Pass the
checkpoint root (or `latest_checkpoint.json`) to `--resume-from`; recovery finds
the newest validated snapshot even when the pointer update was interrupted. An
interruption within an epoch resumes from the latest completed-epoch snapshot
and repeats the incomplete epoch. Keep the same training arguments, dataset
manifest, model/tokenizer revisions, effective precision, and compatible
software/device capabilities. Each snapshot embeds its selected best
checkpoint, so copying the complete snapshot directory to another output root
does not leave the resumed result pointing at the original run's best model.
The saved DataLoader generator state allows the next epoch's shuffle and worker
seeds to continue reproducibly; exact bitwise results across different GPU
hardware or library builds are not guaranteed.

## Validation and held-out test evaluation

Inspect the selected checkpoint pointer, then run evaluation separately for
validation and, after selection is complete, the held-out test:

```powershell
python -m experiment.evaluate_sst2 `
  --checkpoint-dir outputs/milestone2.3/canonical-seed42/checkpoints/checkpoint-step-00001000 `
  --split-dir data/processed/milestone2.3-seed42 `
  --split validation `
  --output-dir outputs/milestone2.3/canonical-seed42/evaluation-validation

python -m experiment.evaluate_sst2 `
  --checkpoint-dir outputs/milestone2.3/canonical-seed42/checkpoints/checkpoint-step-00001000 `
  --split-dir data/processed/milestone2.3-seed42 `
  --split test `
  --output-dir outputs/milestone2.3/canonical-seed42/evaluation-test
```

The step in these commands is an example; use the relative checkpoint path
reported by `best_checkpoint.json`. Evaluation checks the split CSV hash and
source training-file hash against checkpoint provenance, then writes
`validation_metrics.json/.csv` or `test_metrics.json/.csv`. Existing output
files are not overwritten. Use a separate evaluation directory for each run.

## Checkpoint format and safeguards

Each immutable checkpoint directory contains:

```text
metadata.json
model.safetensors
tokenizer/                 # packaged fast-tokenizer files
```

Metadata schema `nlp-defense-sst2`, version 1, records model kind and complete
DistilBERT configuration; model/tokenizer IDs and exact revisions; tokenizer
file hashes; byte mapping (`byte + 1`, padding 0); alignment contract version;
label mapping; sequence and byte limits; model architecture settings; seed,
training configuration, split manifest/source hashes; and best validation
metrics, plus Python, NumPy, pandas, scikit-learn, PyTorch, Transformers,
tokenizers, and safetensors versions. Canonical weights include the complete backbone, byte CNN, scalar
fusion gate and projections, gap embedding/projection, and classifier.

The loader verifies schema, metadata consistency, tokenizer and weight hashes,
model vocabulary, expected caller-supplied identifiers/revisions when given,
and strict state-dict keys/shapes. It loads only the packaged tokenizer with
`local_files_only=True`, restores weights from safetensors, and never falls
back to random weights or another tokenizer. `save_training_state()` is an
optional separate PyTorch artifact for optimizer/scheduler/scaler state;
loading it requires `trusted=True` and is not part of inference.

New resumable snapshots record SHA-256 integrity hashes for `training_state.pt`
and the complete embedded best-checkpoint bundle in the staging metadata before
publication. The resume state’s selected metrics are also checked against the
embedded best-checkpoint metadata. A prior resume snapshot without these hashes
is rejected by the crash-recovery validator rather than accepted without
training-state integrity checking; ordinary inference checkpoints remain
loadable. Continue such an older run with the version that created it or start
a new run. The hash detects incomplete or accidentally modified artifacts; it
is not a cryptographic signature against deliberate replacement of both the
state and its metadata.

## Colab GPU guide

1. Start a Colab runtime with a CUDA GPU. Clone the repository or upload a
   source archive containing the `src/`, `experiment/`, and `data/sst2/`
   directories plus `requirements.txt`.
2. Verify Python and CUDA before and after dependency setup. Colab's installed
   PyTorch/CUDA combination must remain compatible with the pinned requirements;
   the repository has not been validated against a live Colab runtime.
3. Run the split-preparation command above. Save the complete generated split
   directory, including `manifest.json` and exclusion CSVs, to persistent
   storage if the runtime may be reset.
4. Mount Google Drive and set each `--output-dir` to a distinct persistent
   Drive directory. Checkpoint snapshots and logs are written there during the
   run. Use the same data/model/tokenizer settings for both models. Add
   `--allow-hub-download` only if the exact configured revisions are not cached.
5. To resume after an interruption, pass `--resume-from` with the checkpoint
   root, such as `.../checkpoints`. The run output
   directory may already contain only its known config, history, and checkpoint
   entries. Do not resume from an untrusted PyTorch training-state artifact.
6. Use `experiment.evaluate_sst2` for validation during review. After choosing
   the model, run the test evaluation once and preserve its JSON/CSV results
   and split manifest alongside the checkpoint.

For a fresh Colab notebook, these are Bash/IPython cells (run in order). Ensure
the feature branch containing Milestone 2.3 is checked out before running them.

```bash
!git clone https://github.com/Suhaniverma2004/nlp-defense.git
%cd nlp-defense
!git fetch origin implementation/milestone-2.3-reproducible-training
!git checkout -b implementation/milestone-2.3-reproducible-training FETCH_HEAD
from google.colab import drive
drive.mount('/content/drive')
!python -m pip install -r requirements.txt
!python -c "import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()"
!python -m experiment.prepare_milestone23_splits --seed 42 --output-dir data/processed/milestone2.3-seed42
!python -m experiment.train_sst2 --model-kind baseline --split-dir data/processed/milestone2.3-seed42 --output-dir /content/drive/MyDrive/nlp-defense/baseline-seed42 --model-id distilbert-base-uncased --model-revision 12040accade4e8a0f71eabdb258fecc2e7e948be --tokenizer-id distilbert-base-uncased --tokenizer-revision 12040accade4e8a0f71eabdb258fecc2e7e948be --seed 42 --epochs 3 --batch-size 8 --gradient-accumulation-steps 2 --learning-rate 0.00002 --max-expanded-length 128 --max-byte-span-length 512 --validation-interval-steps 1000 --mixed-precision fp16 --device cuda --allow-hub-download
!python -m experiment.train_sst2 --model-kind canonical --split-dir data/processed/milestone2.3-seed42 --output-dir /content/drive/MyDrive/nlp-defense/canonical-seed42 --model-id distilbert-base-uncased --model-revision 12040accade4e8a0f71eabdb258fecc2e7e948be --tokenizer-id distilbert-base-uncased --tokenizer-revision 12040accade4e8a0f71eabdb258fecc2e7e948be --seed 42 --epochs 3 --batch-size 8 --gradient-accumulation-steps 2 --learning-rate 0.00002 --max-expanded-length 128 --max-byte-span-length 512 --validation-interval-steps 1000 --mixed-precision fp16 --device cuda --allow-hub-download
```

Google Drive and other mounted or network filesystems may implement directory
rename, file replacement, caching, and flush semantics differently from a
local filesystem. This code validates the entire staging bundle before calling
`os.replace` and stores staging beside the destination, but cannot promise that
a mounted Drive provider exposes a directory rename atomically or durably after
a runtime, process, or host failure. Keep the source data and, for important
results, retain a second copy of completed runs. The project has not executed
interruption or recovery tests against a live mounted Drive.

To compare completed baseline and canonical runs, validate both metadata
manifests against the same split directory before comparing metrics:

```powershell
python -m experiment.compare_sst2_runs `
  --baseline-checkpoint outputs/milestone2.3/baseline-seed42/checkpoints/checkpoint-step-00000010-ab12cd34 `
  --canonical-checkpoint outputs/milestone2.3/canonical-seed42/checkpoints/checkpoint-step-00000010-ef56ab78 `
  --split-dir data/processed/milestone2.3-seed42
```

The validator checks shared task, dataset, tokenizer, optimization, effective
batch, validation, seed, and effective runtime settings while allowing
architecture-specific settings to differ. It labels the result as a
whole-system comparison: baseline CLS pooling/head and canonical expanded-unit
pooling/byte pathway differ together, so the comparison does not isolate the
causal effect of byte features. The report also records the shared
minimum-validation-loss checkpoint-selection rule and fixed-epoch stopping
rule.

These cells explicitly opt into downloading only the immutable model and
tokenizer revisions shown. Omit `--allow-hub-download` when both revisions are
already cached. The paths are fresh-run locations; if a directory already
contains unrelated files, choose a new run/split directory rather than
overwriting it. For resume, repeat the same training command and add
`--resume-from /content/drive/MyDrive/nlp-defense/canonical-seed42/checkpoints`.

The environment needs Python 3.10+ and the repository's declared packages,
GPU-capable PyTorch when CUDA is used, enough GPU memory for the configured
batch and expanded sequence, the local SST-2 source CSVs, and storage for the
split outputs/checkpoints. This project does not upload data or checkpoints to
Drive automatically.

## Verification performed and limitations

Automated tests use generated CSV fixtures, a local WordLevel fast tokenizer,
and tiny DistilBERT configurations. They cover deterministic stratification,
duplicate/conflict handling, both training paths, CPU operation, validation
metrics, gradients, strict safetensors round trips, metadata rejection, and
offline checkpoint tokenizer loading. They also exercise staged snapshot
recovery, interrupted pointer publication, history replacement, portable best
checkpoints, CPU precision-metadata rejection, comparison-manifest mismatches,
and numerical gradient-accumulation equivalence for three examples at
microbatch size two (tolerance `rtol=1e-5`, `atol=1e-6`). No test requires
Internet access.

The exact cached model/tokenizer revision was loaded locally for inspection,
but no SST-2 training or evaluation experiment was run. CUDA and mounted-Drive
execution have not been tested. The test partition is
held out from the new pipeline but may have been seen by historical legacy
training; local legacy checkpoint provenance is unknown. Upstream licensing,
production tokenizer selection, pretrained/task quality, GPU throughput,
cross-hardware reproducibility, and final held-out results remain unresolved.
