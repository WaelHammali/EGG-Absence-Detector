# EGG-Absence-Detector

Automatic detection of absence-seizure periods from EEG. The current pipeline reads MATLAB timetables, aligns clock-time annotations, labels samples, creates overlapping windows, and extracts time- and frequency-domain features. Model training is intentionally deferred until the dataset report is reviewed.

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

Temporal detection (Parts X–XI): out-of-fold probabilities, the final model, intervals and error analysis:

```bash
python scripts/predict_oof.py          # out-of-fold predictions (RF + all four models), model_comparison.csv, models/rf_final.joblib
python scripts/prepare_app_data.py     # filtered signals, recording metadata, technician events (data/processed/app/)
python scripts/evaluate_detection.py   # threshold, outputs/detected_intervals.csv, figures, docs/PARTS_X_XI.md
python scripts/detect.py --recording 190304A_E [--threshold 0.5]
```

## How to run the app

The app only reads precomputed files, so run the three commands above once (after `build_dataset.py --annotations official`), then:

```bash
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:8050>. Nothing is trained at startup. If a precomputed file is missing, the app shows which command to run instead of failing.

The app has three parts, chosen in the header, and one patient selector in the sidebar:

- **Visualisation**: the data only. Choose the patient and which of the 19 channels to show (the list is grouped by scalp region; remove a channel with its ×, "Show all" brings them all back; stacked in 10-20 order or overlaid). Each curve turns green during a real seizure, a diamond marks each seizure start, and the bottom curve is 1 during a seizure and 0 otherwise. A table lists the real seizures.
- **Prediction**: choose one of the four trained models (Decision Tree, Random Forest, KNN, SVM) and see its probability, its detections against the real seizures, a verdict for this patient (found, missed, false alarms) and the table of real vs predicted intervals. The threshold slider updates everything live; each model starts at its own best threshold.
- **Comparison**: the four models together, with no model to choose. "Selected patient" shows, for the patient in the sidebar, a table (seizures found, missed, false alarms, scores) and a timeline with the real seizures on the first row and one row of detections per model. "All patients" shows the overall scores, a bar chart of F1, recall and precision, and a table of every patient against every model.
- Common controls: page length, gain, page buttons or the ← → keys, jump to previous/next seizure, technician notes, the spectrum of any clicked 2 s window, "Save view as PNG" and "Download intervals CSV".
- The sun/moon button switches between the dark and light themes; `?theme=light&page=prediction&recording=211104B_D` in the URL opens a given state directly.

All probabilities shown are out-of-fold: each recording is scored by a model that never saw it. Fonts (Google Fonts) and icons (Iconify) are loaded from the internet; offline, the app falls back to system fonts and icons are not drawn. Screenshots in `outputs/figures/app/` are produced by `python scripts/screenshot_app.py`.

The provisional counts and feature definition are documented in [`docs/DATASET_PARTS_I_VI.md`](docs/DATASET_PARTS_I_VI.md). `200625A_F` and `210427B_C` are excluded for the reasons recorded in `data/interim/exclusions.csv`.

## Project structure

- `src/io.py`: MATLAB timetable and annotation loading.
- `src/labels.py`: sample-level binary labels.
- `src/preprocessing.py`: band-pass filtering before segmentation.
- `src/detection.py`: window probabilities → seizure intervals, event-level comparison with annotations.
- `src/app_data.py`, `src/app_figures.py`, `assets/app.css`: data access, themed figures and styles for `app.py`.
- `src/annotations.py`: provisional/official annotation resolution driven by `config/annotation_decisions.yaml`.
- `src/segmentation.py`: 2 s windows with 50% overlap.
- `src/features.py`: per-channel and across-channel time/frequency features.
- `scripts/convert_to_excel.py`: labeled Excel and Parquet conversion.
- `scripts/build_dataset.py`: consolidated window feature table and count report.
- `outputs/figures/offset_alignment/`: offset searches and seizure-centered waveform checks.
