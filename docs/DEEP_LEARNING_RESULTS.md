# Deep learning results

Compact CNNs trained directly on the cleaned 19-channel EEG (no hand-made features), trained on a GPU. Training recipe and model definitions: [`scripts/train_cnn.py`](../scripts/train_cnn.py); weights: [`models/cnn/`](../models/cnn/).

- **Nested threshold** (main number): each of the 5 outer folds (same `GroupKFold` as every other model here) is scored with its own threshold, chosen only on 3 training recordings held out from that fold and never seen by the test fold. This is the fair number to compare with the SVM.
- **Smoothed**: the same nested thresholds, but the probability is smoothed first (moving median) and an event, once started at the threshold, survives a dip down to threshold − 0.3 instead of being cut into several false alarms (`src/detection.py`: `smooth_probabilities`, `windows_to_intervals_hysteresis`). Pure post-processing, no retraining.
- **Optimistic threshold**: one threshold chosen on all out-of-fold predictions at once, kept only for reference (this is how the classical models' registry numbers are computed too, so the SVM baseline row below is optimistic in the same way: compare it with the CNNs' optimistic scores for a like-for-like gap, or run `python scripts/evaluate_nested.py` to re-score the classical models with a nested threshold).
- **SzCORE**: event scoring from the [SzCORE framework](https://pypi.org/project/timescoring/) (30 s / 60 s tolerance, events closer than 90 s merged), given for comparison with the literature; not directly comparable to this project's own event F1 (SzCORE merges events closer than 90 s, and several absences occur closer together than that).
- **Window AP**: average precision of the raw per-window probability (threshold-independent), each window scored by a model that never saw its patient.

## Results

Best of plain / smoothed nested scoring per model (bold marks which one won):

| Model | Event F1 | Found (of 93) | False alarms | FA / hour | SzCORE event F1 | Window AP |
|---|---:|---:|---:|---:|---:|---:|
| **SVM · Augmented (channel dropout)** (classical baseline) | **0.901** | 88 | 18 | 2.1 | — | — |
| **ShallowConvNet, amplitude *(smoothed)*** | **0.817** | 77 | 19 | 2.2 | 0.819 | 0.672 |
| ShallowConvNet, augmented | 0.815 | 77 | 21 | 2.4 | 0.819 | 0.672 |
| EEGNet, amplitude+noise+channel_dropout+mirror+time_shift *(smoothed)* | 0.811 | 85 | 32 | 3.7 | 0.841 | 0.668 |
| ShallowConvNet, mixup *(smoothed)* | 0.802 | 78 | 24 | 2.7 | 0.831 | 0.660 |
| ShallowConvNet, mirror *(smoothed)* | 0.798 | 79 | 27 | 3.1 | 0.809 | 0.702 |
| ShallowConvNet, time_mask *(smoothed)* | 0.790 | 77 | 25 | 2.9 | 0.797 | 0.727 |
| ShallowConvNet, time_shift | 0.788 | 77 | 26 | 3.0 | 0.824 | 0.646 |
| ShallowConvNet, plain *(smoothed)* | 0.781 | 76 | 26 | 3.0 | 0.794 | 0.652 |
| ShallowConvNet, noise *(smoothed)* | 0.778 | 76 | 27 | 3.1 | 0.791 | 0.643 |
| ShallowConvNet, noise+time_mask *(smoothed)* | 0.774 | 78 | 32 | 3.7 | 0.800 | 0.704 |
| ShallowConvNet, band_stop *(smoothed)* | 0.774 | 76 | 28 | 3.2 | 0.803 | 0.663 |
| ShallowConvNet, channel_dropout *(smoothed)* | 0.767 | 76 | 30 | 3.4 | 0.785 | 0.657 |

Plain vs. smoothed, each model (same fold thresholds, only the post-processing differs):

| Model | Event F1 plain | Event F1 smoothed | Δ |
|---|---:|---:|---:|
| EEGNet, amplitude+noise+channel_dropout+mirror+time_shift | 0.776 | 0.811 | +0.035 |
| ShallowConvNet, amplitude | 0.811 | 0.817 | +0.006 |
| ShallowConvNet, augmented | 0.815 | 0.811 | -0.004 |
| ShallowConvNet, band_stop | 0.770 | 0.774 | +0.005 |
| ShallowConvNet, channel_dropout | 0.755 | 0.767 | +0.012 |
| ShallowConvNet, mirror | 0.782 | 0.798 | +0.016 |
| ShallowConvNet, mixup | 0.795 | 0.802 | +0.008 |
| ShallowConvNet, noise+time_mask | 0.768 | 0.774 | +0.006 |
| ShallowConvNet, noise | 0.772 | 0.778 | +0.006 |
| ShallowConvNet, plain | 0.774 | 0.781 | +0.007 |
| ShallowConvNet, time_mask | 0.781 | 0.790 | +0.009 |
| ShallowConvNet, time_shift | 0.788 | 0.761 | -0.027 |

Optimistic threshold, for reference (not the number to compare with the SVM):

| Model | Event F1 (optimistic) | Threshold | Found | False alarms |
|---|---:|---:|---:|---:|
| EEGNet, amplitude+noise+channel_dropout+mirror+time_shift | 0.826 | 0.99 | 81 | 24 |
| ShallowConvNet, amplitude | 0.823 | 0.90 | 80 | 23 |
| ShallowConvNet, augmented | 0.842 | 0.98 | 77 | 14 |
| ShallowConvNet, band_stop | 0.797 | 0.96 | 82 | 33 |
| ShallowConvNet, channel_dropout | 0.815 | 0.99 | 78 | 22 |
| ShallowConvNet, mirror | 0.820 | 0.99 | 77 | 20 |
| ShallowConvNet, mixup | 0.823 | 0.94 | 76 | 17 |
| ShallowConvNet, noise+time_mask | 0.797 | 0.99 | 83 | 35 |
| ShallowConvNet, noise | 0.826 | 0.99 | 79 | 21 |
| ShallowConvNet, plain | 0.813 | 0.98 | 79 | 24 |
| ShallowConvNet, time_mask | 0.800 | 0.99 | 81 | 31 |
| ShallowConvNet, time_shift | 0.809 | 0.82 | 82 | 30 |

## Reading

- The best CNN (ShallowConvNet, amplitude, event F1 0.817) does not beat the SVM baseline (0.901), a gap of 0.084. This is the expected outcome: with 93 seizures in 21 patients, raw-signal deep learning is unlikely to beat hand-made amplitude and band-power features on a pattern as regular as the 3 Hz absence rhythm. Its winning score is the *smoothed* one — plain thresholding alone did worse (see the plain vs. smoothed table above).
- **Optimism of the CNNs' own threshold choice** (nested − optimistic event F1): EEGNet, amplitude+noise+channel_dropout+mirror+time_shift -0.050, ShallowConvNet, amplitude -0.012, ShallowConvNet, augmented -0.028, ShallowConvNet, band_stop -0.027, ShallowConvNet, channel_dropout -0.060, ShallowConvNet, mirror -0.038, ShallowConvNet, mixup -0.028, ShallowConvNet, noise+time_mask -0.029, ShallowConvNet, noise -0.054, ShallowConvNet, plain -0.039, ShallowConvNet, time_mask -0.019, ShallowConvNet, time_shift -0.021. A negative value is how much the single-threshold number overstates the model; the same kind of gap the classical models have, and the reason the nested number is the one to trust.
- **Augmentation** (amplitude scaling, noise, channel dropout, left-right mirror, applied to both classes so the seizure share stays near 4 %, unlike the classical models' first augmentation attempt) is compared against the plain CNN when both were run.

Reproduction: `python scripts/export_cnn_data.py`, train on a GPU with `python scripts/train_cnn.py --data data/cnn ...`, copy the outputs to `models/cnn/` and `outputs/predictions/`, then `python scripts/evaluate_cnn.py`. Raw numbers: `outputs/cnn_evaluation.csv`.
