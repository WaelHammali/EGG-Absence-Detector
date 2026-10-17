# Handoff

Written at a session boundary, to pick up work without replaying the full history.

## Project

University data-mining project: detect absence seizures in EEG. 21 recordings, 19 EEG
channels, 93 seizures. Classical pipeline (features + DT/RF/KNN/SVM) is done and solid.
Currently extending with: data augmentation, a redesigned Dash app, and a Kaggle-trained CNN
to compare against the SVM baseline.

## Structure

- `data/processed/recordings/<set>/` and `data/processed/datasets/<name>/` — see
  `docs/DATA_LAYOUT.md`. `src/paths.py` defines every path.
- `scripts/` — one script per pipeline step (`train_registry.py`, `evaluate_cnn.py`,
  `ensemble_predictions.py`, `export_kaggle.py`, `evaluate_nested.py`, ...).
- `kaggle/` — `train_cnn.py` (training script) + `absence_seizure_cnn.ipynb` (notebook,
  generated from `train_cnn.py` via `kaggle/build_notebook.py` — always regenerate the
  notebook after editing `train_cnn.py`, never hand-edit the `.ipynb`).
- `app.py` + `src/app_*.py` — the Dash app (redesigned, "calm clinical" style).
- `docs/` — one report per topic (`MODEL_COMPARISON.md`, `KAGGLE.md`, `ENSEMBLE_RESULTS.md`;
  `DEEP_LEARNING_RESULTS.md` is NOT written yet).

## Key decisions

- 12 classical models: 4 algorithms (DT/RF/KNN/SVM) × 3 training modes
  (unbalanced/balanced/augmented). Best so far: SVM unbalanced, event F1 0.894, 86/93 found,
  16 false alarms, threshold 0.82. This is THE baseline to beat.
- Augmentation (shift + mirror) only helped recall, hurt false alarms overall.
- Ensemble of SVM+RF (no retraining) ties the baseline with 1 fewer false alarm — see
  `docs/ENSEMBLE_RESULTS.md`.
- Deep learning (CNN) on raw signal: first Kaggle run got 0.758-0.801 F1, did not beat SVM —
  expected with only 93 seizures (documented finding, not a failure). Second run in progress
  with tuned hyperparameters.

## Finished

- Full classical pipeline, app redesign, augmented training mode, data folder restructuring
  (`src/paths.py`), Kaggle export/train/evaluate scripts, ensemble script with real results
  committed.

## In progress / next steps

- User is currently re-running `kaggle/absence_seizure_cnn.ipynb` on Kaggle with tuned
  settings (`validation-recordings=5`, `seizure-weight=5`, `min-epochs=5`, `patience=10`)
  after fixing a real `IndexError` bug in `event_counts()`. **Waiting on**: the user to finish
  that Kaggle run and download results.
- Once downloaded: run `python scripts/evaluate_cnn.py --dir <folder>` to produce
  `docs/DEEP_LEARNING_RESULTS.md` (script exists, tested, not yet run on real data).
- `scripts/evaluate_nested.py` (nested-threshold + SzCORE re-evaluation of the 12 classical
  models) was started but is INCOMPLETE — only 2 of 12 models scored
  (`outputs/nested_evaluation.csv`, `docs/NESTED_EVALUATION.md` are partial/stale on disk, NOT
  committed). Needs a full rerun.

## Known issues (fixed)

- `kaggle/train_cnn.py` `event_counts()` had an `IndexError` bug (used pandas groupby index
  LABELS as array POSITIONS) — fixed and verified, commit `b026a70`. If results still look
  wrong, re-check this function first.

## Key commands

- App: `python app.py`
- Rebuild a dataset: `python scripts/build_dataset.py --annotations official`
- Train classical models: `python scripts/train_registry.py`
- Kaggle export: `python scripts/export_kaggle.py`
- After editing `kaggle/train_cnn.py`: `python kaggle/build_notebook.py` (regenerate notebook)
- Score Kaggle CNN results: `python scripts/evaluate_cnn.py --dir outputs/kaggle/<folder>`
- Ensemble: `python scripts/ensemble_predictions.py --models svm_unbalanced rf_unbalanced`
