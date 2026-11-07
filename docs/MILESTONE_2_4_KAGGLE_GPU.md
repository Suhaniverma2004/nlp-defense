# Milestone 2.4 — Kaggle GPU execution

This milestone prepares a Kaggle notebook for the existing Milestone 2.3 SST-2
pipeline. It reuses `src/training.py`, `src/sst2_data.py`,
`src/checkpointing.py`, `src/evaluation.py`, and `src/comparison.py`. It does
not add another trainer, change model architecture, or change dependency pins.
The notebook is a workflow template; it has **not** been executed on Kaggle and
does not report CUDA results.

## Repository baseline and model identity

The notebook source is pinned to merged main commit
`65391a2b0a715bdb67156df4aa1d5725c1edbc76` and checks the runtime packages
before training. A Kaggle source input without Git metadata must include
`MILESTONE_2_4_SOURCE_COMMIT.txt` containing that SHA. The model and tokenizer revision is pinned to
`12040accade4e8a0f71eabdb258fecc2e7e948be` for both
`distilbert-base-uncased` assets. This is the revision characterized and tested
locally in this repository. It is not an independently approved production
revision. The notebook verifies the resolved Hugging Face snapshot identity and required
asset hashes without relying on Transformers private commit-hash fields. It
packages the tokenizer into each checkpoint using the existing checkpoint API.

The requested Kaggle accelerator is NVIDIA T4 x2. The notebook enumerates each
visible CUDA device and records its name, memory capacity, compute capability,
CUDA runtime, PyTorch build, and driver information. All model loading onto the
accelerator, smoke tests, evaluation, and training explicitly select
`cuda:0`. The second T4 is recorded for provenance only. DataParallel,
DistributedDataParallel, and automatic multi-GPU training are not enabled.

## Kaggle paths and source setup

Kaggle input datasets are mounted read-only below `/kaggle/input`; writable
notebook artifacts belong below `/kaggle/working`. The notebook first accepts
an attached repository-source input at `/kaggle/input/nlp-defense-source`.
It verifies the exact source commit using Git metadata or the commit marker.
Otherwise, set the notebook's explicit `ALLOW_REPOSITORY_CLONE = True` option
to clone into `/kaggle/working`; the clone/fetch operation itself reports an
actionable error if the network is unavailable. Cloning defaults to disabled.
The clone fetches the pinned commit and checks it out detached. For an
Internet-disabled session, attach a source dataset containing the repository
tree and commit marker. Never write to `/kaggle/input`.

The split builder reads the repository SST-2 train source and inspects the
repository's original validation source for provenance. It creates the
Milestone 2.3 reproducible splits in a separate `/kaggle/working` directory.
The notebook loads and verifies only the generated train and validation
partitions. It does not load or evaluate the generated held-out test partition.
Use a fresh split-output directory if the source file hashes change; existing
split outputs are never overwritten.

The runtime artifact directory is unique per notebook session and named
`/kaggle/working/nlp-defense-m24-<session-id>`. The split CSVs live in a separate
`/kaggle/working/nlp-defense-m24-splits-<session-id>` directory so the export
cell cannot accidentally package the held-out split or the original dataset.
Only the split manifest, runtime report, experiment
configs, histories, evaluation reports, and complete checkpoint run
directories are exported.

## Dependency profiles and installation policy

The repository pins PyTorch 2.10.0, Transformers 5.2.0, and Tokenizers 0.22.2;
the locally verified environment used those versions. Repository CI runs
Python 3.13 and development documentation also records CPython 3.14.2. Kaggle
maintains separate CPU and GPU notebook images and updates them periodically,
so the actual notebook Python, PyTorch build, CUDA runtime, and package versions
must be recorded from the live session. See the [Kaggle Docker image
repository](https://github.com/Kaggle/docker-python) and [Kaggle package
environment documentation](https://www.kaggle.com/docs/packages).

The **locally validated profile** remains the unchanged `requirements.txt`
environment: Python 3.14.2, PyTorch 2.10.0, Transformers 5.2.0 and Tokenizers
0.22.2, with its resolved packages recorded in checkpoints. The separate
**Kaggle T4 candidate profile** is named
`kaggle-t4x2-py3.13-torch2.11.0+cu130-transformers5.20.2-tokenizers0.23.3-v1`
and verifies the following exact versions before model execution:

| Component | Candidate version |
|---|---:|
| Python | 3.13.16 |
| PyTorch | 2.11.0+cu130 |
| CUDA runtime reported by PyTorch | 13.0 |
| Transformers | 5.20.2 |
| Tokenizers | 0.23.3 |
| NumPy | 2.1.3 |
| pandas | 2.3.3 |
| scikit-learn | 1.6.1 |

This profile is still labeled **candidate**; local validation does not certify
Kaggle execution. The exact check includes the PyTorch CUDA build suffix and
fails closed on every mismatch. It does not install or upgrade packages,
downgrade packages, run `pip install -r requirements.txt`, or replace Kaggle's
CUDA-enabled PyTorch. Do not bypass the check. A newly observed Kaggle image
requires a new reviewed profile or a deliberate profile update, not an
unrecorded exception.

Before full training, run the notebook cells in order: (1) select T4 x2 and
record both devices, capacities, driver and selected `cuda:0`; (2) pass exact
candidate package checks and the CUDA tensor probe; (3) identify the repository
at immutable commit `65391a2b0a715bdb67156df4aa1d5725c1edbc76`; (4) prepare and
verify split provenance while loading only train and validation; (5) resolve
the exact model/tokenizer revision; (6) resolve effective precision and shared
settings; (7) explicitly opt into bounded runtime probes. These probes check
Unicode/emoji alignment, whitespace gaps, literal and structural special
tokens, baseline/canonical forward-backward updates, FP16 autocast and
GradScaler, checkpoint integrity, offline reload/inference, and epoch-boundary
recovery/resume using the existing project APIs. To execute this stage, set
`RUN_CUDA_SMOKE_TESTS=True` in the first notebook code cell while leaving
`RUN_FULL_BASELINE_TRAINING=False` and
`RUN_FULL_CANONICAL_TRAINING=False`. A failed or incomplete probe keeps full
training blocked even if a training opt-in is later set. The profile remains
labeled candidate after the probe run pending review of its evidence.

The exported runtime provenance records the full resolved package inventory,
NVIDIA driver, both GPU identities and memory capacities, selected device,
repository commit, immutable revisions, split manifest and hashes, effective
precision, shared training config and probe results. The manifest includes
held-out split hashes, but this workflow does not load or inspect held-out test
rows. Baseline/canonical comparison remains whole-system only.

The model-assets cell is local-cache-only by default. Set the explicit
`ALLOW_PINNED_ASSET_DOWNLOAD` switch only when Internet access is enabled and
the exact immutable revision is intended to be fetched. No mutable model ref
is accepted. The same switch adds `--allow-hub-download` to both baseline and
canonical CLI invocations; when false, both remain local-cache-only. Checkpoint
reload temporarily switches Hugging Face to offline mode and restores the
previous environment values immediately afterward, so later model loading is
not unintentionally forced offline.

## Execution order

Open `notebooks/milestone_2_4_kaggle_gpu.ipynb` in Kaggle, select GPU T4 x2 in
notebook settings, and run the cells in order. The environment cell records
both GPUs but selects only `cuda:0`. The dependency cell verifies versions and
CUDA operation before any model execution. The source cell uses either the
attached repository input or a clone checked out at the inspected merged-main
commit, not an unpushed feature-branch revision.

The initial controlled training settings are deliberately conservative:
microbatch size 2, gradient accumulation 8, effective batch size 16, maximum
expanded sequence 128, maximum UTF-8 byte span 512, three epochs, seed 42,
AdamW, learning rate 2e-5, warmup ratio 0.1, and validation at epoch boundaries
(`validation_interval_steps=0`). Baseline and canonical configurations share
all task, data, optimizer, scheduling, precision, device, and validation
settings. Their architecture-specific behavior remains different by design.

The baseline uses DistilBERT CLS pooling and its classifier head; the canonical
model mean-pools expanded token/gap units and includes the byte pathway. Any
comparison is **whole-system only**. It does not isolate the causal effect of
byte features.

The CUDA smoke-test switch defaults to false and, when enabled, trains each
model on at most four train and four validation examples. It records optimizer
updates, finite validation metrics, elapsed time, peak memory, and checkpoint
reload/inference. A separate bounded check simulates interruption after an
epoch checkpoint is published but before its latest-pointer update, then
validates recovery and continues from the recovered epoch boundary. Smoke
metrics are engineering checks, not research results.

Full baseline and canonical training each have a separate false-by-default
opt-in variable. Run them only after the bounded checks pass and a persistent
artifact plan is ready. The two cells call `python -m experiment.train_sst2`
with matched shared settings, then evaluate the selected checkpoint on the
validation split only through `src.evaluation.evaluate_checkpoint_split`. The
comparison cell calls the existing comparison manifest validator. Full
training subprocess output is retained under the artifact root's `logs/`. The
notebook contains no test-split evaluation command.

## GPU memory and mixed precision

The notebook requests the configured mixed-precision mode, resolves the
effective dtype through `src.training.resolve_mixed_precision`, and runs a CUDA
autocast probe before passing the same request to training. It records the
effective mode from `TrainingResult`; it does not infer FP16 support from the
T4 label alone. If the probe or training reports a different effective mode,
review the saved metadata before comparing runs.

If CUDA runs out of memory, stop the active cell and resume from the latest
completed epoch snapshot. Start with microbatch size 2 and accumulation 8. If
needed, reduce the microbatch to 1 and increase accumulation to 16 for **both**
models, retaining effective batch size 16. Keep other shared settings fixed,
use the same changed config for both runs, and start new output directories.
If sequence or byte-span overflow occurs, inspect offending inputs and use the
same explicit length limits for both models; do not silently truncate source
text. `torch.cuda.empty_cache()` may release unused cached blocks but cannot
free tensors that are still referenced.

## Checkpoint recovery and artifact persistence

The existing trainer saves validation-selected model bundles and complete
epoch-boundary resume snapshots. Bundles include Safetensors model weights,
metadata, packaged tokenizer assets and integrity hashes. Resume snapshots
also include optimizer, scheduler, scaler, RNG, data-loader generator, epoch,
global step, history, and an embedded copy of the selected best checkpoint.
Keep the entire run directory intact when resuming; pass its
`checkpoints/latest_checkpoint.json` (or `checkpoints` root) to
`--resume-from`. Use the same dataset manifest, immutable model/tokenizer
revisions, shared training config, effective precision, and compatible
software/device environment.

The export cell makes a ZIP beneath `/kaggle/working` containing complete run
directories and the provenance/configuration/logging files, but not the SST-2
CSV data. Download the ZIP from the Kaggle output pane or save the notebook
version with its output files. For longer-lived storage, upload the ZIP as a
private Kaggle Dataset or copy it to another user-managed store after verifying
its contents. Kaggle sessions and working storage are not guaranteed to
survive resets; neither this workflow nor this documentation assumes persistent
storage. Do not put Kaggle API tokens, Hugging Face tokens, or other credentials
in notebook cells or exported artifacts.

## Verification status and remaining Kaggle checks

Local validation covers notebook syntax and profile logic only. It does not
validate the candidate against Kaggle hardware. The following checks remain
to be run in the actual Kaggle T4 x2 session before full training:

1. Pass the exact candidate package and driver/GPU inventory checks; confirm
   both T4s and selected `cuda:0`.
2. Confirm the pinned fast-tokenizer/model revision resolves and run Unicode,
   emoji, whitespace/gap and special-token alignment probes.
3. Run baseline and canonical CUDA forward/backward/optimizer probes and the
   FP16 autocast/GradScaler probe; inspect finite values and device placement.
4. Save and integrity-check both checkpoints, reload them offline, perform
   inference, and recover/resume completed epoch snapshots.
5. Review the saved runtime and split provenance. Only then consider enabling
   matched baseline/canonical full training.

No actual Kaggle model training, checkpoint compatibility, CUDA training,
Kaggle persistence, full training, test-set evaluation, performance
improvement, or adversarial robustness result is claimed until these checks
are run and reviewed.


## Incremental validation evidence and training authorization

The repository setup cell creates `validation_report.json` before environment checks and before any candidate probe. The report maintains append-only validation runs; every mandatory probe begins as `pending`, is persisted as `running`, then becomes `passed` or `failed`. A failure record contains the probe name, exception type, concise message, UTC timestamp, and SHA-256 validation-context fingerprint. Exceptions are re-raised, and the report is not a performance-results file. A successful export includes both `validation_report.json` and its current status in `runtime_report.json` and `workflow_config.json`. If a reported probe fails, the notebook attempts to write `runtime_report.json` and a failure snapshot ZIP before re-raising. The incremental report also remains in its session artifact directory if export is unavailable.

The context fingerprint uses canonical sorted-key JSON and SHA-256. It binds the exact candidate profile and installed package inventory, GPU/device/CUDA/driver metadata, repository commit and relevant source-tree digest, immutable model/tokenizer identifiers and revisions, the dataset split manifest and hashes, and effective training settings (including sequence bounds, microbatch/effective batch size, accumulation, optimizer, scheduler, precision, epoch count, and validation interval). It excludes transient paths and timestamps. Both baseline and canonical training gates rebuild this context from the current session and require it to match the active report run and every mandatory probe result. A changed or incomplete context requires a fresh smoke validation.

This fingerprint detects stale or mismatched execution evidence; it does not prove that the Kaggle runtime is trustworthy, that filesystem writes survive power loss, or that training results are scientifically valid. JSON replacement uses a same-directory temporary file and `os.replace` where supported. Atomic visibility and durability depend on the filesystem; mounted/network storage and sudden power loss may not provide the same guarantees.

## Execution-context binding and smoke authorization

The candidate notebook builds one authoritative execution context immediately before validation. It recomputes the exact shared training settings, resolves the effective autocast dtype, checks cached `SHARED`/precision/dtype values, and verifies both the `torch.device("cuda:0")` object and active CUDA index. Any stale notebook variable is rejected before a probe can start. The context fingerprint includes the current source-tree digest, exact runtime/package and GPU/driver metadata, repository commit, immutable model/tokenizer revisions, dataset/split manifest, complete `TrainingConfig` defaults and shared settings, effective precision, and the explicitly bounded smoke overrides.

Both baseline and canonical smoke runs construct their `TrainingConfig` from the validated context. The only smoke overrides are two epochs and four examples per split; each is represented in the fingerprint, and the output directory is treated as a non-semantic location. The notebook checks the constructed dataclass against the expected context immediately before training. CUDA scaler/autocast uses that same resolved dtype and device. Each full-training path recomputes the current context at its gate and requires a complete successful report with the identical fingerprint. Editing settings, source, data manifests, package/runtime metadata, model revisions, or device invalidates prior evidence.

This deterministic fingerprint establishes equality of the recorded inputs; it does not prove the underlying software, hardware, dataset labels, or experiment are scientifically valid. It cannot detect unrecorded external state or guarantee filesystem durability. Kaggle execution is still required to establish candidate-runtime compatibility.

## Immutable source pairing

The notebook checks out repository source commit `65391a2b0a715bdb67156df4aa1d5725c1edbc76`. This commit contains `src/kaggle_runtime.py` and the training, alignment, checkpoint, evaluation, and experiment modules imported by the notebook. If source is attached without Git metadata, `MILESTONE_2_4_SOURCE_COMMIT.txt` must contain this exact SHA. The notebook's source-tree digest reads only files resolved under the selected `REPO_ROOT`; inputs that resolve outside that checkout are rejected. It does not read sibling local worktrees. The corrective notebook revision can therefore use this already merged source commit without depending on the corrective PR's future commit.

## Immutable Hugging Face asset verification

Notebook preflight uses the supported `huggingface_hub.snapshot_download` API with both repository IDs and revisions supplied explicitly. It verifies that the returned cache path identifies `models--distilbert-base-uncased/snapshots/12040accade4e8a0f71eabdb258fecc2e7e948be`, that model and tokenizer resolve to the same snapshot, and that `config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`, and model weights are present and hashable. Optional tokenizer metadata files are included in the verified asset hashes when present. The existing `AutoTokenizer.from_pretrained` and `AutoConfig.from_pretrained` calls remain pinned to the same IDs/revision and honor the same explicit local-cache/download switch.

The resolved repository identity, revision, and required asset hashes are included in the validation fingerprint and exported `runtime_report.json`/`workflow_config.json`. Training continues to pass the pinned revisions to `DistilBertModel.from_pretrained` and `AutoTokenizer.from_pretrained`; checkpoint metadata continues recording those revisions. Notebook verification does not rely on Transformers private `_commit_hash` fields. Runtime compatibility and the cache layout still require confirmation in the actual Kaggle image; local tests use synthetic cache snapshots and do not download model assets.
