# Handoff

State of the project at the end of a long session, to continue without replaying the history.

## Where things stand

- Classical pipeline (features + DT/RF/KNN/SVM × unbalanced/balanced/augmented = 12 models),
  event-level detection, the Dash app (Home, Visualisation, Prediction, Comparison) and the
  reports in `docs/` are finished.
- **Best single model:** SVM unbalanced, event F1 0.894 (86/93 found, 16 false alarms, threshold
  0.82). The threshold was chosen on the same out-of-fold predictions it is scored on, so this
  number is slightly optimistic.
- **Best blend:** SVM + ShallowConvNet, event F1 0.901 (90/93 found, 19 false alarms) with the
  threshold chosen on the other folds only (`docs/ENSEMBLE_RESULTS.md`). The gain over the SVM is
  small (4 more seizures found, 3 more false alarms) and was not tested for noise.
- **CNNs** (`models/cnn/`, `docs/DEEP_LEARNING_RESULTS.md`): ShallowConvNet augmented 0.815,
  EEGNet augmented 0.776, ShallowConvNet plain 0.774 (nested threshold). None beats the SVM;
  expected with 93 seizures in 21 patients. Smoothing + hysteresis helps only EEGNet (0.776 → 0.811).
- The cloud-notebook tooling was removed in the cleanup commit (recoverable from the commit before it).
  Training recipe and model classes are in `scripts/train_cnn.py`.

## Open items

- `scripts/evaluate_nested.py` (nested threshold + SzCORE for the 12 classical models) was only run
  for 3 models; its cache is in `outputs/nested/` (git-ignored). Running it fully (about an hour,
  memory-heavy: it was killed once for lack of RAM) would give the SVM a nested score for a fair
  comparison with the CNNs' nested scores.
- Ask the neurologist to review the false alarms (especially `210504B_C` and `210427B_C`):
  some are probably real, unannotated seizures, which would raise every model's precision.
- The CNN loading example in `models/cnn/README.md` was written but not run (no PyTorch here).
- `data/excel/` (2 GB, labeled Excel copy of each recording) is not used by any later step; it can
  be deleted and rebuilt with `python scripts/convert_to_excel.py`.

## Known bug history

- `event_counts()` in `scripts/train_cnn.py` once used pandas group labels as array positions and
  crashed on sliced tables; fixed (commit `b026a70`). If CNN scores look wrong, check it first.

## Commands

See `CLAUDE.md`. Commit messages in this repo: plain text, no `Co-Authored-By` trailer.
