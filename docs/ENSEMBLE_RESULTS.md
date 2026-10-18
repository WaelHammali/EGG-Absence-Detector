# Ensemble results

Blends of out-of-fold probabilities from several already-trained models (no retraining): research report, part A7, "the cheapest way to get a real gain". Built with `python scripts/ensemble_predictions.py`.

- **held-out-fold** (main number): for each outer fold, the threshold is swept on the other 4 folds' predictions only, never on the fold being scored — valid even though nothing was retrained for this fold.
- **optimistic**: one threshold swept on all out-of-fold predictions at once, kept for reference (the same convention as the registered models' own best threshold).

## Results

| Ensemble | Event F1 (held-out-fold) | Found (of 93) | False alarms | FA / hour | SzCORE event F1 |
|---|---:|---:|---:|---:|---:|
| **SVM · Augmented (channel dropout)** (single model, baseline) | **0.901** | 88 | 18 | 2.1 | — |
| SVM · Unbalanced + cnn_shallow_augmented | 0.901 | 90 | 19 | 2.2 | 0.900 |
| SVM · Augmented (channel dropout) + cnn_shallow_amp | 0.896 | 89 | 19 | 2.2 | 0.892 |
| SVM · Unbalanced + Random Forest · Unbalanced | 0.894 | 85 | 15 | 1.7 | 0.894 |
| SVM · Unbalanced + Random Forest · Unbalanced + KNN · Unbalanced | 0.874 | 83 | 18 | 2.1 | 0.881 |
| SVM · Unbalanced + KNN · Unbalanced | 0.871 | 81 | 15 | 1.7 | 0.872 |
| SVM · Unbalanced + cnn_eegnet_augmented | 0.868 | 84 | 19 | 2.2 | 0.864 |
| SVM · Unbalanced + cnn_shallow_plain | 0.860 | 80 | 15 | 1.7 | 0.850 |
| 2×SVM · Unbalanced + Random Forest · Unbalanced | 0.849 | 83 | 23 | 2.6 | 0.870 |
| SVM · Unbalanced + Random Forest · Unbalanced + cnn_eegnet_augmented | 0.839 | 82 | 24 | 2.7 | 0.848 |
| SVM · Augmented (channel dropout) + Random Forest · Unbalanced | 0.823 | 84 | 31 | 3.5 | 0.866 |
| SVM · Augmented (channel dropout) + Random Forest · Augmented (channel dropout) | 0.817 | 81 | 29 | 3.3 | 0.846 |

## Reading

- **SVM · Unbalanced + cnn_shallow_augmented** ties the baseline (0.901 vs 0.901), with 90 found and 19 false alarms against the baseline's 88 found and 18. A tie with a different error profile can still be worth reporting, but is not on its own a reason to switch the default model.

Reproduction: `python scripts/ensemble_predictions.py --models <key> <key> ... --extra <path.parquet> ...` (see the script's docstring). Raw numbers: `outputs/ensemble_evaluation.csv`.
