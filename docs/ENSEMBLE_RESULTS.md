# Ensemble results

Blends of out-of-fold probabilities from several already-trained models (no retraining): research report, part A7, "the cheapest way to get a real gain". Built with `python scripts/ensemble_predictions.py`.

- **held-out-fold** (main number): for each outer fold, the threshold is swept on the other 4 folds' predictions only, never on the fold being scored — valid even though nothing was retrained for this fold.
- **optimistic**: one threshold swept on all out-of-fold predictions at once, kept for reference (the same convention as the registered models' own best threshold).

## Results

| Ensemble | Event F1 (held-out-fold) | Found (of 93) | False alarms | FA / hour | SzCORE event F1 |
|---|---:|---:|---:|---:|---:|
| **SVM · Unbalanced** (single model, baseline) | **0.894** | 86 | 16 | 1.8 | — |
| SVM · Unbalanced + Random Forest · Unbalanced | 0.894 | 85 | 15 | 1.7 | 0.894 |
| SVM · Unbalanced + Random Forest · Unbalanced + KNN · Unbalanced | 0.874 | 83 | 18 | 2.1 | 0.881 |
| SVM · Unbalanced + KNN · Unbalanced | 0.871 | 81 | 15 | 1.7 | 0.872 |
| 2×SVM · Unbalanced + Random Forest · Unbalanced | 0.849 | 83 | 23 | 2.6 | 0.870 |

## Reading

- **SVM · Unbalanced + Random Forest · Unbalanced** ties the baseline (0.894 vs 0.894), with 85 found and 15 false alarms against the baseline's 86 found and 16. A tie with a different error profile can still be worth reporting, but is not on its own a reason to switch the default model.

Reproduction: `python scripts/ensemble_predictions.py --models <key> <key> ... --extra <path.parquet> ...` (see the script's docstring). Raw numbers: `outputs/ensemble_evaluation.csv`.
