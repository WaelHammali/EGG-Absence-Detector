# CLAUDE.md

## Project

University data-mining project: detect absence seizures in EEG recordings. 21 recordings, 19
EEG channels, 93 seizures. Pipeline: filter signal → window → extract features → train
classical models (DT/RF/KNN/SVM, 3 training modes) → evaluate detection event by event →
Dash app to explore results. CNNs trained on a GPU machine are compared with them, and blends of models are scored too.

## Structure

- `data/processed/recordings/<set>/`, `data/processed/datasets/<name>/` — one folder per
  recording set / dataset; see `docs/DATA_LAYOUT.md`. `src/paths.py` defines every path used
  by the scripts — add new paths there, not hardcoded.
- `scripts/` — one script per pipeline step, each independently runnable.
- `models/` — `registry.json` + the 12 classical models; `models/cnn/` holds the CNN weights
  (the model classes live in `scripts/train_cnn.py`, which is self-contained so it can be copied
  to a GPU machine).
- `app.py` + `src/app_*.py` — the Dash app.
- `docs/` — one Markdown report per topic.

## Commands

- Run the app: `python app.py`
- Rebuild the official dataset: `python scripts/build_dataset.py --annotations official`
- Train the 12 classical models: `python scripts/train_registry.py`
- Package data for GPU training: `python scripts/export_cnn_data.py` → `data/cnn/` (patient data, never commit)
- Score the CNNs: `python scripts/evaluate_cnn.py` (reads `models/cnn/` and `outputs/predictions/`)
- Blend already-trained models: `python scripts/ensemble_predictions.py --models <key> <key>`

## Commit messages

Plain text only — no `Co-Authored-By` trailer, no "Generated with Claude Code" line, for every
commit in this repo.
