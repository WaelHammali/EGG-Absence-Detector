# Deep learning results

Compact CNNs trained directly on the cleaned 19-channel EEG (no hand-made features), trained on a GPU. Training recipe and model definitions: [`scripts/train_cnn.py`](../scripts/train_cnn.py); weights: [`models/cnn/`](../models/cnn/).

- **Nested threshold** (main number): each of the 5 outer folds (same `GroupKFold` as every other model here) is scored with its own threshold, chosen only on 3 training recordings held out from that fold and never seen by the test fold. This is the fair number to compare with the SVM.
- **Smoothed**: the same nested thresholds, but the probability is smoothed first (moving median) and an event, once started at the threshold, survives a dip down to threshold − 0.3 instead of being cut into several false alarms (research report, part A5; `src/detection.py`: `smooth_probabilities`, `windows_to_intervals_hysteresis`). Pure post-processing, no retraining.
- **Optimistic threshold**: one threshold chosen on all out-of-fold predictions at once, kept only for reference (this is how the classical models' registry numbers are computed too, so the SVM baseline row below is optimistic in the same way: compare it with the CNNs' optimistic scores for a like-for-like gap, or run `python scripts/evaluate_nested.py` to re-score the classical models with a nested threshold).
- **SzCORE**: event scoring from the [SzCORE framework](https://pypi.org/project/timescoring/) (30 s / 60 s tolerance, events closer than 90 s merged), given for comparison with the literature; not directly comparable to this project's own event F1 (SzCORE merges events closer than 90 s, and several absences occur closer together than that).
- **Window AP**: average precision of the raw per-window probability (threshold-independent), each window scored by a model that never saw its patient.

## Results

Best of plain / smoothed nested scoring per model (bold marks which one won):

| Model | Event F1 | Found (of 93) | False alarms | FA / hour | SzCORE event F1 | Window AP |
|---|---:|---:|---:|---:|---:|---:|
| **SVM · Unbalanced** (classical baseline) | **0.894** | 86 | 16 | 1.8 | — | — |
| **ShallowConvNet, augmented** | **0.815** | 77 | 21 | 2.4 | 0.819 | 0.672 |
| EEGNet, augmented *(smoothed)* | 0.811 | 85 | 32 | 3.7 | 0.841 | 0.668 |
| ShallowConvNet, plain *(smoothed)* | 0.781 | 76 | 26 | 3.0 | 0.794 | 0.652 |

Plain vs. smoothed, each model (same fold thresholds, only the post-processing differs):

| Model | Event F1 plain | Event F1 smoothed | Δ |
|---|---:|---:|---:|
| EEGNet, augmented | 0.776 | 0.811 | +0.035 |
| ShallowConvNet, augmented | 0.815 | 0.811 | -0.004 |
| ShallowConvNet, plain | 0.774 | 0.781 | +0.007 |

Optimistic threshold, for reference (not the number to compare with the SVM):

| Model | Event F1 (optimistic) | Threshold | Found | False alarms |
|---|---:|---:|---:|---:|
| EEGNet, augmented | 0.826 | 0.99 | 81 | 24 |
| ShallowConvNet, augmented | 0.842 | 0.98 | 77 | 14 |
| ShallowConvNet, plain | 0.813 | 0.98 | 79 | 24 |

## Reading

- The best CNN (ShallowConvNet, augmented, event F1 0.815) does not beat the SVM baseline (0.894), a gap of 0.079. This matches the research report's expectation: with 93 seizures in 21 patients, raw-signal deep learning is unlikely to beat hand-made amplitude and band-power features on a pattern as regular as the 3 Hz absence rhythm.
- **Optimism of the CNNs' own threshold choice** (nested − optimistic event F1): EEGNet, augmented -0.050, ShallowConvNet, augmented -0.028, ShallowConvNet, plain -0.039. A negative value is how much the single-threshold number overstates the model; the same kind of gap the classical models have, and the reason the nested number is the one to trust.
- **Augmentation** (amplitude scaling, noise, channel dropout, left-right mirror, applied to both classes so the seizure share stays near 4 %, unlike the classical models' first augmentation attempt) is compared against the plain CNN when both were run.

Reproduction: `python scripts/export_cnn_data.py`, train on a GPU with `python scripts/train_cnn.py --data data/cnn ...`, copy the outputs to `models/cnn/` and `outputs/predictions/`, then `python scripts/evaluate_cnn.py`. Raw numbers: `outputs/cnn_evaluation.csv`.
