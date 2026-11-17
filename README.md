# DP-ST-GCN: Dual-Person Skeleton Action Recognition

A reconstructed PyTorch implementation of the methodology in **Interaction-Aware Skeleton-Based Action Recognition Using a Dual-Person Spatial-Temporal Graph Convolutional Network**, based on the supplied 18-page manuscript, Sections 3–3.1.

**Status:** Newly written code, not original author scripts. Original training data, split lists, checkpoints and logs were not provided. The paper's reported accuracies and timing have **not** been reproduced. Artificial demo data only check software execution. Details absent or ambiguous in the paper are recorded in [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md).

## Included

- Fixed 25-node or 50-node symmetric normalized skeleton graphs.
- Six spatial-temporal blocks: `3 → 64 → 64 → 128 → 128 → 256 → 256`; temporal kernel 9; stride 2 in blocks 3 and 5.
- Temporal attention: `256 → 64 → 1`, softmax over time, weighted pooling, dropout and 51-class head.
- Joint, bone, joint-motion and bone-motion representations.
- Shared-backbone independent-person baseline with elementwise max of person logits.
- Unified dual-person model and an additional same-size graph without cross-person edges.
- Missing-aware normalization, uniform resampling and synchronized augmentation.
- Adam, cosine schedule, inverse-square-root class weights, label smoothing, CUDA AMP and early stopping.
- Probability fusion with validation-only lambda selection.
- Prediction archives, macro/per-class metrics, confusion matrices, attention plots, partial-observation evaluation and direct single/two-stream timing.
- Split leakage checks, unit tests and a GitHub Actions CPU workflow.

## Installation

Use Python 3.10–3.12. From the repository root:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell alternative: .venv\Scripts\Activate.ps1
```

Install an appropriate PyTorch build for your machine. For a CPU demonstration:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
python -m unittest discover -s tests -v
```

This reconstruction uses the modern `torch.amp` API and declares PyTorch >=2.3. It does not claim to recreate the manuscript's PyTorch 2.1/CUDA 11.8 environment. For exact archival reproduction, freeze a validated environment and record your GPU/runtime versions.

## Quick smoke test (artificial data)

```bash
python scripts/make_demo.py
python scripts/train.py --manifest data/demo/manifest.json --config configs/demo.json --stream joint --output outputs/demo_joint --device cpu
python scripts/evaluate.py --manifest data/demo/manifest.json --checkpoint outputs/demo_joint/best.pt --output outputs/demo_eval --device cpu
python scripts/plot_results.py --predictions outputs/demo_eval/predictions.npz --output outputs/demo_plots
```

The demo has all 51 labels but no meaningful learning task. It uses only 20 sampled frames and one epoch for speed; real-data defaults use 100 frames. On machines with large CPU thread pools, set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` before a small smoke test to reduce overhead.

## Prepare real PKU-MMD data

Obtain the dataset and official protocol files through the [PKU-MMD project](https://struct002.github.io/PKUMMD/). Confirm the release/phase and its cross-subject partition. The paper does not supply its actual split lists; this repository does not guess them or silently make a random clip-level split.

Create `data/recordings.json`, with one entry per continuous recording/view:

```json
[
  {
    "skeleton": "raw/0001-L.skeleton",
    "labels": "raw/0001-L.label",
    "recording": "0001",
    "subjects": ["ACTUAL_SUBJECT_ID_A", "ACTUAL_SUBJECT_ID_B"],
    "split": "train"
  }
]
```

Paths are relative to `recordings.json`. Enter actual participant IDs, including both people in interactions. `recording` must identify the source event shared by its different camera views; all views and clips from that event stay in one split. Provide all train/validation/test entries. The code rejects overlapping subject IDs or source recordings across splits. These checks validate the metadata you supply; they cannot establish that invented or incorrect metadata match the official protocol.

Each skeleton row must contain 150 numeric values arranged as person-major, joint-major XYZ, reshaped into `(frames,2,25,3)`. Confirm layout and joint indexing against your release. Label rows must contain `class,start,end,confidence`; the converter accepts comma- or whitespace-separated values. Original class IDs are assumed 1–51 and converted to 0–50. Endpoints are treated as inclusive. Explicitly choose the actual zero- or one-based frame convention:

```bash
python scripts/prepare_data.py --recordings data/recordings.json --output data/processed --frame-base 1 --minimum-confidence 1
```

The resulting manifest points to variable-length `.npy` clips. If you already have clips, create that same manifest schema directly. Do not redistribute dataset files in this code repository.

## Verify protocol settings

Before training real data, review `configs/model.json` and the implementation notes. Confirm:

- Dataset phase/release, official cross-subject split and validation/test role.
- Joint ordering, coordinate layout, missing-value convention and frame indexing.
- Exact nine cross-person pairs and normalization convention.
- Maximum epoch count, random seed and augmentation/stream conventions.

The provided nine cross joints are an **explicit provisional interpretation**, not recovered original settings. `protocol_verified` defaults to `false`; set it to `true` only after replacing or confirming those settings. Real-data training will otherwise stop with an explanation. Demo training is separately marked synthetic.

## Train models

```bash
python scripts/train.py --manifest data/processed/manifest.json --config configs/model.json --stream joint --mode dual --output outputs/dual_joint
python scripts/train.py --manifest data/processed/manifest.json --config configs/model.json --stream bone --mode dual --output outputs/dual_bone
python scripts/train.py --manifest data/processed/manifest.json --config configs/model.json --stream joint --mode single --output outputs/single_joint
python scripts/train.py --manifest data/processed/manifest.json --config configs/model.json --stream joint --mode dual_no_cross --output outputs/dual_no_cross
```

Use `--stream joint_motion` or `--stream bone_motion` for motion experiments. Change and record the seed for repeated-run estimates. Class weights use training labels only. Validation accuracy selects the checkpoint; patience is 12 epochs. Class weights are normalized to mean one. Every class must occur in training. The default 100 maximum epochs is a new configurable choice because the paper does not specify it.

The no-cross model retains 50 nodes while removing interaction edges; it is an additional suggested control, not a claim that the manuscript reported this experiment. The single-versus-dual comparison changes graph layout and prediction pooling and does not isolate interaction edges alone.

## Evaluate and fuse

```bash
python scripts/evaluate.py --checkpoint outputs/dual_joint/best.pt --manifest data/processed/manifest.json --split val --output outputs/val_joint
python scripts/evaluate.py --checkpoint outputs/dual_bone/best.pt --manifest data/processed/manifest.json --split val --output outputs/val_bone
python scripts/fuse.py --joint outputs/val_joint/predictions.npz --bone outputs/val_bone/predictions.npz --select-on-validation --output outputs/val_fused
```

The fusion script aligns clip IDs and verifies labels, graph mode and observation fraction. It searches joint weight lambda from 0 to 1 in steps of 0.01, breaking ties toward the smallest lambda. This grid is an implementation choice. Outputs from the tuning set remain **validation results**, not an independent estimate.

For an untouched test partition, first evaluate both checkpoints using `--split test`, then apply the fixed validation weight:

```bash
python scripts/fuse.py --joint outputs/test_joint/predictions.npz --bone outputs/test_bone/predictions.npz --lambda-file outputs/val_fused/fusion_weight.json --output outputs/test_fused
```

If you use the official held-out set for checkpoint/lambda selection, label it validation; do not subsequently call its results independent test performance. A separate subject-disjoint validation subset of official training subjects is a defensible new protocol but would differ from unspecified original settings. Preserve that distinction in reporting.

## Attention, prefixes and timing

```bash
python scripts/plot_results.py --predictions outputs/val_joint/predictions.npz --output outputs/plots
python scripts/partial_observation.py --checkpoint outputs/dual_joint/best.pt --manifest data/processed/manifest.json --split val --output outputs/prefix_joint --device cpu
python scripts/benchmark.py --checkpoint outputs/dual_joint/best.pt --device cpu --output outputs/timing_joint.json
python scripts/benchmark.py --checkpoint outputs/dual_joint/best.pt --bone-checkpoint outputs/dual_bone/best.pt --joint-weight 0.5 --device cpu --output outputs/timing_fused.json
```

Prefixes are truncated before normalization and resampling so future frames cannot affect the observed input. These experiments assume known action boundaries and are not online onset detection. For fused prefixes, evaluate each stream at the same fraction and apply the saved validation lambda with `fuse.py`; never retune on each test prefix.

Attention corresponds to temporally downsampled backbone positions, not an explanation of causal frame importance. Classwise attention arrays are exported; interaction-class IDs must be verified rather than guessed. Evaluation timing is batched amortized model time. `benchmark.py` reports direct batch-1 timings, including both forwards and probability fusion when two checkpoints are supplied. Both exclude skeleton estimation, preprocessing, device transfers and acquisition. They are not robot end-to-end latency.

## Repository structure

| Path | Purpose |
|---|---|
| `dpstgcn/graph.py` | Anatomy tree and normalized graph |
| `dpstgcn/preprocess.py` | Normalization, augmentation and streams |
| `dpstgcn/model.py` | Backbone, attention and independent-person baseline |
| `dpstgcn/data.py` | Dataset loader and leakage checks |
| `scripts/prepare_data.py` | Segmentation and manifest creation |
| `scripts/train.py` | Training and checkpoint selection |
| `scripts/evaluate.py` | Predictions, metrics and attention |
| `scripts/fuse.py` | Validation selection and fixed-weight fusion |
| `scripts/partial_observation.py` | Prefix evaluation sweep |
| `scripts/plot_results.py` | Attention and confusion plots |
| `scripts/benchmark.py` | Single/two-stream batch-1 timing |
| `tests/` | Graph, preprocessing, model and leakage checks |

Generated checkpoints contain local training state. Load only trusted checkpoints. Predictions and participant metadata remain local by default. No software license is assigned; the authors should select a license and add final manuscript citation details before inviting reuse. Do not claim this repository is official author code unless the authors adopt and validate it.

Suggested GitHub repository name: **`dp-st-gcn-interaction-recognition`**. Upload the extracted directory's contents, including hidden `.github/` and `.gitignore`, while leaving real datasets and generated outputs out. No training weights or research results are bundled.
