# EGG-Absence-Detector

Automatic detection of absence-seizure periods from EEG. The current pipeline reads MATLAB timetables, aligns clock-time annotations, labels samples, creates overlapping windows, and extracts time- and frequency-domain features. Four algorithms are then trained in three training modes and compared on seizure detection; a Dash app shows the signals, the predictions and the comparison.

## Data flow

```text
data/raw/*.mat + Annotations.xlsx
        ↓ clock alignment and reviewed annotations
data/interim/offsets.csv + annotations_clean.csv
        ↓ labeled conversion
data/excel/*.xlsx + data/processed/<recording>.parquet
        ↓ 2 s windows, 50% overlap, feature extraction
data/processed/windows_features.parquet
```

The 19 EEG channels of the 10-20 montage are used (Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2), selected by name; every annotated recording has all of them. ECG, EMG and SLI are excluded. EEG amplitudes and feature arrays are stored as `float32`. The earlier 8-channel results are kept in `outputs/archive_8ch/` and compared in [`docs/CHANNELS_8_VS_19.md`](docs/CHANNELS_8_VS_19.md) (`python scripts/compare_channel_sets.py`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the original MAT files and `Annotations.xlsx` in `data/raw/`. Raw files are read-only inputs.

## Commands

Inspect the source files:

```bash
python scripts/inspect_data.py
```

Reproduce offset inference and the special ambiguous-recording review:

```bash
python scripts/align_clock_offsets.py
python scripts/review_210427_alignment.py
```

Create labeled Excel and per-recording Parquet files, then build the feature dataset:

```bash
python scripts/convert_to_excel.py
python scripts/build_dataset.py
```

Official annotations with preprocessing (default: zero-phase Butterworth band-pass 0.5–40 Hz, order 4, applied per channel to each full recording before segmentation), then Part VII and Parts VIII–IX:

```bash
python scripts/build_dataset.py --annotations official                      # filtered → windows_features_official.parquet
python scripts/build_dataset.py --annotations official --preprocess none \
    --skip-conversion                                                        # unfiltered → windows_features_official_raw.parquet
python scripts/plot_filtering.py                                             # one seizure before/after filtering
python scripts/analyze_features.py --annotations official --preprocess none  # unfiltered reference first
python scripts/analyze_features.py --annotations official                    # filtered report + comparison
python scripts/train_models.py                                               # docs/PARTS_VIII_IX.md
```

`--skip-conversion` reuses the labeled per-recording Parquet files after checking their labels against the current annotations. Reports: [`docs/DATASET_PARTS_I_VI_OFFICIAL.md`](docs/DATASET_PARTS_I_VI_OFFICIAL.md), [`docs/PART_VII_ANALYSIS_OFFICIAL.md`](docs/PART_VII_ANALYSIS_OFFICIAL.md), [`docs/PARTS_VIII_IX.md`](docs/PARTS_VIII_IX.md).

Temporal detection (Parts X–XI, Random Forest) and the 12 selectable models:

```bash
python scripts/predict_oof.py          # Part X: out-of-fold Random Forest predictions, models/rf_final.joblib
python scripts/build_augmented.py      # extra seizure windows for the augmented mode, in their own file
python scripts/train_registry.py       # 12 models = DT, RF, KNN, SVM x unbalanced, balanced, augmented [--modes augmented]
python scripts/prepare_app_data.py     # filtered signals, reference channels, metadata, technician events, comparison curves
python scripts/evaluate_detection.py   # Part X-XI: threshold, outputs/detected_intervals.csv, figures, docs/PARTS_X_XI.md
python scripts/compare_models.py       # docs/MODEL_COMPARISON.md
python scripts/detect.py --recording 190304A_E [--algo dt|rf|knn|svm] [--mode unbalanced|balanced|augmented] [--threshold 0.5]
```

`train_registry.py` writes `outputs/predictions/<algo>_<mode>.parquet` (out-of-fold probabilities, GroupKFold by recording), `models/<algo>_<mode>.joblib` (final models on all 21 recordings) and `models/registry.json` (features, best threshold and scores of each model, default model). Balanced training undersamples the training folds only (all seizure windows + as many normal windows, 5 seeds). Augmented training keeps the whole training fold and adds extra seizure windows of the training recordings only: windows shifted by 0.25, 0.5 and 0.75 s and the left-right mirror of every seizure window (Fp1↔Fp2, F7↔F8, ...). These 8,230 windows are built by `build_augmented.py` into `data/processed/windows_features_official_augmented.parquet`; the official dataset is not changed. Test data always keeps the real proportions and the original windows. `--modes` retrains only the given modes and keeps the other models of the registry. The model files are not in git (the KNN one is 72 MB): rebuild them with the script.

## How to run the app

The app only reads precomputed files, so run `train_registry.py` and `prepare_app_data.py` once (after `build_dataset.py --annotations official`), then:

```bash
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:8050>. Nothing is trained at startup. If a precomputed file is missing, the app shows which command to run instead of failing.

The app opens in a light theme; the moon/sun button in the header switches to the dark theme. It opens on a **Home** page: what the app is for, an example of a real seizure with the best model's detection, then, further down, the seven steps from raw EEG to detected seizures (recordings, filtering, windows, features, the training modes, testing on unseen patients, thresholds) with the numbers read from the data and the registry, the best model and buttons to the other parts. Then three parts, chosen with the tabs in the header, and one patient selector in the sidebar (with the patient's sex, age, recording duration and number of real seizures).

- **Visualisation**: the data only. The 19 channels are toggle chips grouped by scalp region (click a channel to hide or show it; "Show all" brings them all back), stacked in 10-20 order or overlaid. Each curve turns green during a real seizure, a diamond marks each seizure start, and the bottom curve is "yes" during a seizure and "no" otherwise. The plot fills the window: on a 1920×1080 screen the 19 channels, the bottom row and the navigator are visible together without scrolling.
- **Prediction**: the model chosen in the header (Algorithm: DT / RF / KNN / SVM, Training mode: Unbalanced / Balanced / Augmented). A summary line gives the verdict for this patient (found, missed, false alarms) and, on the right, the model's scores on all patients. The plot adds the model's probability, its detections and the threshold line. The threshold slider is in the sidebar; each model starts at its own best threshold and everything updates live.
- **Comparison**: read from top to bottom. *Who wins*: the best of the compared models in a larger card with the reason in one sentence, then the others in rank order. *Why*: event F1 against the threshold, precision-recall curves and the training class distribution. *Detail*: sortable ranking with the best values highlighted, confusion matrices, a timeline of the selected patient (real seizures + one row per model) and event F1 per patient. "Same training mode", "Same algorithm, all modes", "All models" or any custom list choose what is compared. One color per algorithm; solid = unbalanced, dashed or hatched = balanced, dotted = augmented.
- Plot toolbar: previous/next page (also the ← → keys), previous/next seizure, **Spectrum** (the frequency content of a 2 s window; it also opens when you click a curve), **Intervals** (table of the real seizures, with the model's detections in Prediction), "Save view as PNG" and "Download intervals CSV". Spectrum and Intervals open in a side panel so the plot keeps its place.
- Sidebar view controls: stacked / overlay, page length (10 s, 30 s, 60 s, full), gain, technician notes, and reference channels (ECG, EMG, SLI drawn in gray, labeled "reference — not used by the model"; they are never features).
- `?theme=dark&page=prediction&recording=211104B_D` in the URL opens a given state directly.

All probabilities shown are out-of-fold: each recording is scored by a model that never saw it. Fonts (Google Fonts) and icons (Iconify) are loaded from the internet; offline, the app falls back to system fonts and icons are not drawn. Screenshots in `outputs/figures/app/` (1920×1080, 1440×900 and 1280×800, both themes) are produced by `python scripts/screenshot_app.py`; `outputs/figures/app/before/` keeps the previous design for comparison. The design is described in [`docs/design/REDESIGN_PROPOSAL.md`](docs/design/REDESIGN_PROPOSAL.md).

The provisional counts and feature definition are documented in [`docs/DATASET_PARTS_I_VI.md`](docs/DATASET_PARTS_I_VI.md). In the provisional build `200625A_F` and `210427B_C` are excluded for the reasons recorded in `data/interim/exclusions.csv`; the official build includes `210427B_C`.

## Project structure

- `src/io.py`: MATLAB timetable and annotation loading.
- `src/labels.py`: sample-level binary labels.
- `src/preprocessing.py`: band-pass filtering before segmentation.
- `src/detection.py`: window probabilities → seizure intervals, event-level comparison with annotations.
- `src/app_data.py`, `src/app_figures.py`, `src/app_home.py`, `assets/app.css`: data access, themed figures, the Home page and styles for `app.py`.
- `src/annotations.py`: provisional/official annotation resolution driven by `config/annotation_decisions.yaml`.
- `src/segmentation.py`: 2 s windows with 50% overlap.
- `src/features.py`: per-channel and across-channel time/frequency features.
- `scripts/convert_to_excel.py`: labeled Excel and Parquet conversion.
- `scripts/build_dataset.py`: consolidated window feature table and count report.
- `outputs/figures/offset_alignment/`: offset searches and seizure-centered waveform checks.
