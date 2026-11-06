# Milestone 2.4 — Kaggle GPU execution

This milestone prepares a Kaggle notebook for the existing Milestone 2.3 SST-2
pipeline. It reuses `src/training.py`, `src/sst2_data.py`,
`src/checkpointing.py`, `src/evaluation.py`, and `src/comparison.py`. It does
not add another trainer, change model architecture, or change dependency pins.
The notebook is a workflow template; it has **not** been executed on Kaggle and
does not report CUDA results.

## Repository baseline and model identity

The notebook is based on the inspected merged Milestone 2.3 main commit
`efa0a1ac2aa5dfbccaa8e14b17a193b71e0713e2` and checks the runtime packages
before training. A Kaggle source input without Git metadata must include
`MILESTONE_2_4_SOURCE_COMMIT.txt` containing that SHA. The model and tokenizer revision is pinned to
`12040accade4e8a0f71eabdb258fecc2e7e948be` for both
`distilbert-base-uncased` assets. This is the revision characterized and tested
locally in this repository. It is not an independently approved production
revision. The notebook verifies the resolved commit metadata and packages the
tokenizer into each checkpoint using the existing checkpoint API.

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
Otherwise it clones the repository into `/kaggle/working`, fetches the pinned
commit, and checks it out detached when Kaggle Internet access is enabled. For
an Internet-disabled session, attach a source dataset containing the repository
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

## Dependency compatibility and installation policy

The repository pins PyTorch 2.10.0, Transformers 5.2.0, and Tokenizers 0.22.2;
the locally verified environment used those versions. Repository CI runs
Python 3.13 and development documentation also records CPython 3.14.2. Kaggle
maintains separate CPU and GPU notebook images and updates them periodically,
so the actual notebook Python, PyTorch build, CUDA runtime, and package versions
must be recorded from the live session. See the [Kaggle Docker image
repository](https://github.com/Kaggle/docker-python) and [Kaggle package
environment documentation](https://www.kaggle.com/docs/packages).

The notebook fails before model execution if Python is below 3.10, CUDA is
unavailable, or the required PyTorch/Transformers/Tokenizers versions do not
match the tested pins. It prints all observed pinned-package versions. It does
not install or upgrade packages automatically. Do not run `pip install -r
requirements.txt` over Kaggle's live GPU image: that can replace the Kaggle
PyTorch/CUDA build and alter transitive packages. If a pin differs, stop and
review compatibility first. The safe setup is fail-closed: record Kaggle's
Python, CUDA-enabled PyTorch build, CUDA runtime, and GPU inventory; select a
runtime compatible with the repository's tested package pins; and never
reinstall PyTorch from the generic requirements file. If only the Transformers
stack differs, prepare and review a version-resolved installation plan against
the already installed PyTorch/CUDA build, install only that reviewed package
set, restart the kernel, then rerun package, import, `pip check`, and CUDA
smoke checks. This notebook intentionally does not automate a Kaggle package
installation because the live image version has not been verified here.

The model-assets cell is local-cache-only by default. Set the explicit
`ALLOW_PINNED_ASSET_DOWNLOAD` switch only when Internet access is enabled and
the exact immutable revision is intended to be fetched. No mutable model ref
is accepted. Checkpoint reload switches Hugging Face to offline mode and loads
the tokenizer and weights only from the checkpoint bundle.

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

Local validation covers the notebook JSON structure and Python cell syntax.
The existing project tests and CPU environment do not validate a Kaggle GPU
runtime. The following checks remain to be run in the actual Kaggle T4 x2
session before full training:

1. Confirm runtime Python and exact PyTorch, Transformers, and Tokenizers
   versions; confirm the PyTorch CUDA build imports successfully.
2. Confirm two T4 devices are visible, record both memory capacities, and
   verify all model parameters and batches stay on `cuda:0`.
3. Confirm pinned model/tokenizer revisions resolve and their commit metadata
   matches the requested SHA.
4. Run both bounded CUDA smoke tests, offline checkpoint reloads, and copied
   epoch-boundary recovery; inspect peak memory and effective precision.
5. Run matched baseline/canonical training only after the smoke and recovery
   checks, inspect the shared-settings manifest, and archive the artifacts.

No CUDA execution, Kaggle persistence, full training, test-set evaluation,
performance improvement, or adversarial robustness result is claimed by this
milestone.
