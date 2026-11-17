# Methodology mapping and unresolved details

This implementation follows the supplied manuscript's six-block network and equations 1–7. 

## Mapping

| Manuscript | Code |
|---|---|
| Action segmentation; 100 uniform frames | `scripts/prepare_data.py`, `prepare_clip` |
| First-person root centering and common scaling | `prepare_clip` |
| Fixed normalized adjacency, Eq. 1 | `adjacency` |
| Spatial projection/graph propagation, Eq. 2 | `Block.forward` |
| Attention and weighted pooling, Eq. 3 | `Backbone.forward` |
| Within-person child-minus-parent bones, Eq. 4 | `representation` |
| Probability fusion, Eq. 5 | `scripts/fuse.py` |
| Independent person max pooling, Eq. 6 | `Recognizer.forward` |
| Weighted smoothed cross-entropy, Eq. 7 | `scripts/train.py` |

## Explicit reconstruction choices

1. **Nine cross edges:** The manuscript lists “spine base, spine mid, neck (left and right), left elbow, right elbow, left wrist, and right wrist,” but this does not unambiguously identify nine distinct Kinect joints. Provisional zero-based list `[0,1,2,4,8,5,9,6,10]` means spine base, spine mid, neck, both shoulders, both elbows and both wrists. This includes shoulders as an explicit interpretation, not a fact recovered from the paper. Set the verified actual list before real training.
2. **Graph:** A standard 25-joint Kinect-v2 rooted tree is encoded with one undirected edge per anatomical link. Each node gets exactly one self-loop before degree normalization. The prose is unclear whether A already has loops; adding loops twice would change Eq. 1, so this code avoids it. No spatial partitions, adaptive adjacency or edge-importance masks are added.
3. **Scale:** The code uses the Euclidean distance between first-person temporal-mean neck and root coordinates, using frames where both are valid. It does not use a mean of framewise distances. The wording permits both interpretations; confirm the original implementation. Scale below 1e-6 or missing person-A root frames are rejected rather than silently imputed or swapping identities.
4. **Missing data:** Raw all-zero/nonfinite joint coordinates are marked invalid before normalization. Their mask is preserved after centering, since a valid root becomes zero. All-zero coordinates cannot distinguish a genuine origin from a missing raw joint without confidence metadata; verify the source format. Unavailable child/parent pairs create zero bones. Missing people remain zero through augmentation and motion. The converter rejects nonfinite raw coordinates; direct NPY input masks them, except that a missing root still causes rejection.
5. **Uniform sampling:** Round-to-nearest indices from a linear grid including endpoints; repeated indices fill short clips. The original tie/rounding convention is not specified. Prefixes are selected on the raw clip before scale calculation and resampling.
6. **Augmentation:** Shared Y-axis rotation and scale for both people; independent valid-joint Gaussian noise; synchronized five-frame masking with probability 0.5. Features are derived after joint augmentation. Root jitter, mask ordering and motion derivative direction are not fully specified in the paper.
7. **Motion:** Forward difference `x[t+1]-x[t]`; last frame zero; differences touching missing joints are zero. Motion-stream construction is not defined mathematically in the supplied text, so this is explicit rather than presumed original behavior.
8. **Residual and dropout:** Spatial linear projection without bias, BN/ReLU/dropout, temporal convolution without bias and BN, residual addition then ReLU. Residual projections include BN. The classifier also uses dropout 0.5. These complete underspecified implementation details. No input batch-normalization layer is invented.
9. **Baseline:** Shared backbone processes both person branches in a single batch, so BN uses shared branch statistics. Max pooling acts on logits, not softmax probabilities; the manuscript does not resolve this. A fully missing person is excluded from max pooling. Its zero branch still enters backbone BN during training, an explicit batch-statistics convention. No person identity swaps are introduced.
10. **Loss:** Inverse-square-root training-class counts normalized to mean one; PyTorch weighted cross-entropy with label smoothing 0.1 and its standard weighted mean reduction. Equation 7 does not define the reduction denominator. Logged losses are sample-weighted averages of batch-reduced losses, not a recomputation of global weighted loss. No validation labels enter class weights.
11. **Training:** Maximum 100 epochs and seed 20261117 are new defaults because missing in the paper. Scheduler steps once per epoch. Strict validation-accuracy improvement selects the checkpoint; ties count toward patience. Checkpoints save optimizer/scheduler state for audit but no resume interface is claimed. CUDA AMP is used only when CUDA is selected. Seeding and deterministic cuDNN settings reduce randomness but do not guarantee identical results across hardware/runtime versions.
12. **Fusion:** Arithmetic probability fusion, lambda grid step 0.01 and smallest-lambda tie break. The original selected lambda/search resolution are not supplied. Clip-ID alignment prevents accidental row-order fusion. A validation-tuned score is labeled validation performance.
13. **Metrics:** Macro precision/recall/F1 average over all 51 classes with zero for undefined values. Absent evaluation classes are explicit in per-class support. Confusion matrices are normalized by true-class row. Per-class attention is exported without inventing interaction-class IDs.
14. **Efficiency:** Actual parameter counts are calculated from the reconstructed network rather than forced to the reported 1.7 million. A mismatch is informative and must be reconciled with the original implementation. FLOPs are not estimated by an undocumented counting rule. Timing scripts distinguish model-only batch-1 latency from batched throughput and robot-level latency.

## Dataset/protocol discrepancy

The [official PKU-MMD description](https://struct002.github.io/PKUMMD/) identifies Phase I as 51 classes, partitioned into 41 daily and 10 interaction actions; the supplied manuscript instead describes 45 individual and six interaction classes. Phase II has 41 classes. Verify the dataset release and exact interaction subset before subgroup reporting. The package does not invent an interaction-class list or silently select a phase.

Use official protocol metadata to assign participants and source recordings. Never randomly split clips or camera views from the same event across partitions. If official held-out data are used to select checkpoints/fusion weights, those results are validation and require separate data for independent testing. Clip classification with annotated boundaries differs from the benchmark's continuous action-detection task; these scripts do not compute temporal detection mAP or demonstrate robot deployment.

## Required for an exact reproduction claim

Obtain the original release checksum, source/split metadata, actual nine correspondences, joint tree, preprocessing rules, interaction-class IDs, maximum epochs, seed(s), selected lambda, checkpoint and logs. Run full real-data experiments, repeat seeds, and reconcile parameter counts and metrics before describing this code as reproducing published results. No published values are hard-coded to manufacture agreement.
