# CLAUDE.md

## Project

University data-mining project: detect absence seizures in EEG recordings. 21 recordings, 19
EEG channels, 93 seizures. Pipeline: filter signal → window → extract features → train
classical models (DT/RF/KNN/SVM, 3 training modes) → evaluate detection event by event →
Dash app to explore results. A Kaggle-trained CNN is being added for comparison.

## Structure

- `data/processed/recordings/<set>/`, `data/processed/datasets/<name>/` — one folder per
  recording set / dataset; see `docs/DATA_LAYOUT.md`. `src/paths.py` defines every path used
  by the scripts — add new paths there, not hardcoded.
- `scripts/` — one script per pipeline step, each independently runnable.
- `kaggle/` — `train_cnn.py` is the source of truth; `absence_seizure_cnn.ipynb` is generated
  from it via `python kaggle/build_notebook.py`. Never hand-edit the `.ipynb`.
- `app.py` + `src/app_*.py` — the Dash app.
- `docs/` — one Markdown report per topic.

## Commands

- Run the app: `python app.py`
- Rebuild the official dataset: `python scripts/build_dataset.py --annotations official`
- Train the 12 classical models: `python scripts/train_registry.py`
- Export data for Kaggle: `python scripts/export_kaggle.py`
- Score Kaggle CNN results: `python scripts/evaluate_cnn.py --dir outputs/kaggle/<folder>`
- Blend already-trained models: `python scripts/ensemble_predictions.py --models <key> <key>`

## Commit messages

Plain text only — no `Co-Authored-By` trailer, no "Generated with Claude Code" line, for every
commit in this repo.
