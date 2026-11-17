# Software validation

Validation performed on CPU with Python 3.12, PyTorch 2.15.0+cpu and inputs.

Passed:

- Seven unit tests: graph normalization/connectivity, missing-data feature streams, raw-prefix leakage, preservation of person separation, forward/backward computation for three graph modes, subject leakage rejection and prediction-ID alignment.
- One-epoch joint and bone training on artificial 51-class data.
- Single-stream evaluation and metric exports.
- Validation lambda selection and joint–bone probability fusion.
- Attention/confusion plotting and per-class attention export.
- Prefix evaluation at 50% and 100%, truncating raw data before preprocessing.
- Direct batch-1 two-stream timing with both model forwards and arithmetic probability fusion.
- Converter fixture with comma/whitespace annotations and inclusive frame endpoints.
- Python syntax compilation.

The reconstructed single-stream model has **1,750,707 trainable parameters** with 51 classes. Two independent streams have **3,501,414 parameters**. These are calculated counts, not values forced to match Table 3. Synthetic accuracy and CPU timing do not validate the paper's findings or GPU latency and are intentionally not bundled as research results.

No real PKU-MMD files, author checkpoints, subject splits or reported results were available for verification. Real-data training is gated on review of ambiguous protocol settings. The synthetic pipeline uses fewer frames and epochs than real-data defaults for speed. See IMPLEMENTATION_NOTES.md for all reconstruction decisions.
