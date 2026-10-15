# Training the deep learning models on Kaggle

The classical models (DT, RF, KNN, SVM) run fine on this laptop's CPU. The compact CNNs
(`kaggle/train_cnn.py`) need a GPU to train in a reasonable time, so they run on Kaggle's free
GPU quota (about 30 GPU-hours/week) instead. This page is the step-by-step for that.

## 1. Build and upload the data package (once)

```bash
python scripts/export_kaggle.py
```

Writes `data/kaggle/eeg-absence/` (~800 MB, git-ignored): the 19 filtered EEG channels and
per-sample labels of each of the 21 recordings, the 31,419 official windows with their `fold`
(1–5, the same GroupKFold used everywhere else), and the annotations. Only recording codes —
no patient names or dates.

Edit `data/kaggle/eeg-absence/dataset-metadata.json` first: replace `YOUR_KAGGLE_USERNAME`
with your Kaggle username.

**Upload it as a PRIVATE dataset** (this is patient EEG — keep it private, and check with your
professor that uploading it to Kaggle is allowed):

- **Web UI**: kaggle.com → *Create* → *New Dataset* → drag the `eeg-absence/` folder →
  make sure visibility is **Private** → *Create*.
- **Or the CLI**, if you have a Kaggle API token (`~/.kaggle/kaggle.json`):
  ```bash
  pip install kaggle
  cd data/kaggle/eeg-absence
  kaggle datasets create -p . --dir-mode zip
  ```

## 2. Open the notebook

Upload `kaggle/absence_seizure_cnn.ipynb` to Kaggle (*Create* → *New Notebook* → *File* →
*Import Notebook*), or copy its cells into a new one. It is self-contained: the first code
cell writes out `train_cnn.py`, so you only need this one file plus the dataset from step 1.

In the notebook's settings (right-hand panel):

- **Accelerator**: GPU T4 x2 (or P100) — not "None", or training runs on CPU and is slow.
- **Internet**: off (not needed; keeps the run reproducible).
- **Add Input** → your `eeg-absence` dataset.

## 3. Run it

Cells, in order:

1. Checks the GPU and finds the dataset under `/kaggle/input/`.
2. Writes `train_cnn.py`.
3. A ~1 minute quick check (one fold, one seed, two epochs) to catch mistakes early.
4. **ShallowConvNet, plain** — the first real result, no augmentation.
5. **ShallowConvNet, augmented** — amplitude scaling, noise, channel dropout, left-right
   mirror, small time shift, applied to *both* classes (so the seizure share stays near 4 %,
   unlike the first classical-model augmentation, which only added seizure windows).
6. **EEGNet, augmented** — a second architecture; skip this cell if short on GPU time, the
   two ShallowConvNet runs already give a fair comparison point.
7. A summary table against the SVM baseline (event F1 0.894, 86/93 found, 16 false alarms).
8. Zips `/kaggle/working/cnn_outputs/` for download.

Each full run (5 outer folds × 3 seeds, early stopping) takes roughly 30–90 minutes on a T4.
Kaggle notebooks keep running if you close the tab (commit the version, or use *Save & Run
All*), but a session still times out after 12 hours.

## 4. Bring the results back

Download the zip from the notebook's *Output* tab, then:

```bash
mkdir -p outputs/kaggle
unzip ~/Downloads/cnn_outputs.zip -d outputs/kaggle/
```

Each run writes, under `outputs/kaggle/`:

- `predictions/cnn_<arch>_<plain|augmented>.parquet` — out-of-fold probability of every
  window, same format as `outputs/predictions/<algo>_<mode>.parquet`;
- `predictions/..._validation.parquet` — probabilities of the 3 held-out training recordings
  per fold, used to choose each fold's threshold without ever looking at its test fold;
- `models/..._fold<k>_seed<s>.pt` and `..._final.pt` (+ `.onnx` when export succeeds) — weights;
- `..._config.json` / `..._history.json` — hyperparameters, fold thresholds, training curves.

Then score them exactly like the other 12 models (nested threshold, SzCORE, added to
`docs/NESTED_EVALUATION.md` and the app's Comparison page):

```bash
python scripts/evaluate_cnn.py
```

(`scripts/evaluate_cnn.py` is the next step — not written yet. Say the word when you have
real Kaggle results to bring in, and that's what reads `outputs/kaggle/` and wires them in.)

## Keeping the notebook in sync

`kaggle/absence_seizure_cnn.ipynb` embeds the full text of `kaggle/train_cnn.py` in one cell.
If you edit `train_cnn.py`, regenerate the notebook rather than editing the `.ipynb` by hand:

```bash
python kaggle/build_notebook.py
```
