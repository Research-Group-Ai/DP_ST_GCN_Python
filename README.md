# DP-ST-GCN research implementation

This is an independent, paper-guided PyTorch implementation, not author-provided source code. The original PKU-MMD raw video/skeleton data and official subject split are **not** included. Convert each segmented action clip into `.npy` containing a float32 array of shape `(T,2,25,3)` using the Kinect v2 joint order, and create a CSV manifest with `path,label,split,subject`. Labels must be integer IDs 0–50; `split` must be `train` or `val`. Paths are relative to the CSV directory. Preserve the official subject-disjoint protocol and sample order across streams.

## Install
`pip install -r requirements.txt`

## Train four streams
```
python train.py --manifest manifest.csv --stream joint --output joint.pt
python train.py --manifest manifest.csv --stream bone --output bone.pt
python train.py --manifest manifest.csv --stream joint_motion --output joint_motion.pt
python train.py --manifest manifest.csv --stream bone_motion --output bone_motion.pt
python train.py --manifest manifest.csv --stream joint --baseline --output baseline.pt
python train.py --manifest manifest.csv --stream joint --no-cross --output no_cross.pt
```

## Evaluate and fuse
```
python evaluate.py --manifest manifest.csv --checkpoint joint.pt --output joint_val.npz
python evaluate.py --manifest manifest.csv --checkpoint bone.pt --output bone_val.npz
python fusion.py --joint joint_val.npz --bone bone_val.npz --output fused_val.npz
python plots.py --predictions fused_val.npz --history joint.pt.history.json
python benchmark.py
```

## Notes
- This reproduces the described architecture, graph, streams, augmentation, training, and evaluation at an implementation level; no paper accuracy is guaranteed.
- PKU-MMD raw data extraction, segment annotations, and cross-subject subject-ID assignment depend on the dataset release and must be supplied by the researcher.
- The paper does not fully specify jitter standard deviation, temporal-mask duration, fusion lambda, or exact coordinate normalization; those are explicitly chosen implementation defaults.
- Fusion lambda is tuned on validation; report that result as validation accuracy only. Fix lambda before evaluating an independent test set.
- The original paper's baseline max fusion is implemented on logits. Additional no-cross graph ablation is included.
- Partial observation masks future frames after feature construction, matching the paper's masked-frame description; this is not continuous online detection.

## Generate paper Figures 1–10
The new plotting scripts distinguish schematic figures and paper-reported reference values from measurements. They never fabricate unreported training histories or class-level measurements.

```
python plot_architecture.py --output-dir figures
python plot_all_figures.py --output-dir figures --history joint.pt.history.json --predictions fused_val.npz
python partial_observation.py --manifest manifest.csv --checkpoints joint.pt bone.pt joint_motion.pt bone_motion.pt --output partial_observation.csv
python plot_all_figures.py --output-dir figures --history joint.pt.history.json --predictions fused_val.npz --partial partial_observation.csv
```

Optional Figure 8 class-specific attention: evaluate each desired interaction class on a separately filtered manifest containing only that class (preserve the val split), then pass `--attention handshake=handshake.npz giving_object=giving.npz pushing=pushing.npz` to `plot_all_figures.py`. Note that this requires class-filtered predictions and that model attention is not a causal explanation.

`plot_all_figures.py` without datasets can produce only paper-reported reference Figures 5 and 10. Figure 4 requires saved training history, Figures 6–7 require predictions, Figure 8 requires class-specific attention arrays, and Figure 9 requires partial-observation evaluation. Paper values are not independent reproduction results.

The partial-observation experiment zeros unseen input frames *after* features are computed. For motion inputs this is an offline masking proxy, not an online causal motion pipeline. The implementation's independent baseline applies max pooling to logits. The paper does not provide raw per-class values or its exact 51-class indexing mapping.
