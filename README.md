# EEG Absence Detector

Automatic detection of absence seizures in EEG recordings, built for a university data-mining project.

The pipeline reads the MATLAB recordings, aligns the neurologist's annotations, filters the signal, cuts it into 2 s windows and describes each window with 320 features. Four algorithms (Decision Tree, Random Forest, KNN, SVM) are trained in three ways of handling the class imbalance, then evaluated on patients they never saw. A Dash app shows the signals, the predictions of each model and the comparison of the 12 models.

## Results at a glance

| | |
|---|---|
| Recordings | 21 (8.7 hours), 256 Hz, 19 EEG channels |
| Annotated seizures | 93 |
| Windows | 31,419, of which 1,244 are seizure windows (3.96 %) |
| Features per window | 320 |
| Models | 4 algorithms × 3 training modes = 12 |
| Best model | **SVM, unbalanced**: event F1 0.894, 86 of 93 seizures found, 16 false alarms (1.8 per hour), threshold 0.82 |

Event F1 at each model's best threshold (scores on unseen patients):

| Algorithm | Unbalanced | Balanced | Augmented |
|---|---:|---:|---:|
| SVM | **0.894** | 0.821 | 0.862 |
| Random Forest | **0.867** | 0.785 | 0.854 |
| KNN | **0.828** | 0.519 | 0.692 |
| Decision Tree | **0.290** | 0.148 | 0.171 |

Training on the real proportions and choosing the threshold gives the best result for every algorithm. Undersampling throws away most of the normal windows, and augmentation repeats the same 93 seizures; in both cases the models predict seizures too often on real data. The full analysis (in French) is in [`docs/MODEL_COMPARISON.md`](docs/MODEL_COMPARISON.md).

## Method

### 1. Data

- 22 MATLAB recordings; 21 are used. `200625A_F` has no annotations and only 8 EEG channels.
- The 19 EEG channels of the 10-20 system, selected by name: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2.
- ECG, EMG and photic stimulation (SLI) are shown in the app as references and are never used as features.
- Seizure times come from the official annotation workbook (`Annotations MAJ.xlsx`): `start = DebCE − DebutTrace`. The decisions on special cases are recorded in [`config/annotation_decisions.yaml`](config/annotation_decisions.yaml) and [`docs/ANNOTATION_DECISIONS.md`](docs/ANNOTATION_DECISIONS.md). Raw files are never modified.

### 2. Preprocessing

Each channel of each full recording is filtered with a zero-phase Butterworth band-pass filter, 0.5–40 Hz, order 4 (`scipy.signal.sosfiltfilt`). It removes the electrode drift and the 50 Hz mains noise and keeps the 3 Hz seizure rhythm. The unfiltered dataset is kept for comparison (`windows_features_official_raw.parquet`).

### 3. Windows and labels

The signal is cut into 2 s windows with a 1 s step (50 % overlap). A window is a seizure window (class 1) when at least 50 % of its samples are inside an annotated seizure. This gives 31,419 windows, of which 1,244 are seizure windows (3.96 %).

### 4. Features

16 features per channel:
- 8 in time: mean, std, variance, min, max, amplitude, rms, energy;
- 8 from the spectrum: absolute and relative power of the delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz) and beta (13–30 Hz) bands.

With the average of each feature over the channels: 16 × (19 + 1) = 320 features.

### 5. Training modes

The three modes change only the training data. Test data always keeps the real windows and the real 4 %.

| Mode | Training data of a fold | Seizure share |
|---|---|---:|
| Unbalanced | All training windows, as recorded (about 25,100) | 4 % |
| Balanced | All seizure windows + as many normal windows drawn at random (random undersampling, about 2,000 windows). Repeated with 5 seeds (0–4); seed 0 is saved. | 50 % |
| Augmented | All training windows + extra seizure windows of the training recordings (about 31,700) | 24 % |

The augmented windows are cut again from the filtered signal, never invented:
- **Shifted windows**: windows starting 0.25, 0.5 and 0.75 s after the regular ones, kept only if at least 50 % of the window is seizure (3,493 windows).
- **Left-right mirror** of every seizure window, original and shifted: Fp1↔Fp2, F7↔F8, F3↔F4, T3↔T4, C3↔C4, T5↔T6, P3↔P4, O1↔O2; Fz, Cz and Pz stay in place (1,244 + 3,493 windows). This is valid because absence seizures are generalised, so they appear on both sides of the head.

That makes 8,230 added windows, stored apart in `data/processed/windows_features_official_augmented.parquet`; the official dataset is not changed. A build check confirms that the regular windows reproduce the official dataset exactly.

### 6. Validation

`GroupKFold` with 5 folds by recording: each recording is predicted by a model trained on the other folds, so every score and every probability in the app comes from a model that never saw that patient. Neighbouring windows overlap, so a random split would leak information between training and test.

### 7. From windows to seizures

1. Windows with a probability at or above the threshold are joined into detections.
2. Detections less than 2 s apart are merged, and detections shorter than 2 s are dropped.
3. A real seizure overlapped by a detection is *found*, a seizure with no detection is *missed*, and a detection that touches no seizure is a *false alarm*.

For each model, the threshold is the one between 0.05 and 0.99 that gives the highest event F1. It is chosen on the same out-of-fold predictions, so the scores at the best threshold are slightly optimistic, in the same way for every model. Accuracy is not used: a model that never detects anything is already 96 % accurate.

### Main limits

21 recordings and 93 seizures, default hyperparameters. The missed seizures are mostly in the two oldest patients (17 and 25 years). About 41–46 % of the RF and SVM false alarms fall during hyperventilation, which is used to provoke seizures. Recording `210427B_C` shows several seizure-like bursts with only one annotated, so some "false alarms" may be real seizures.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the original MAT files and `Annotations.xlsx` in `data/raw/`, and the updated workbook `Annotations MAJ.xlsx` at the project root (its path is set in `config/annotation_decisions.yaml`). Raw files are read-only inputs. `data/` and the model files (`models/*.joblib`) are not in git; the commands below rebuild them.

## Commands

### Inspection and annotation alignment

```bash
python scripts/inspect_data.py
python scripts/align_clock_offsets.py
python scripts/review_210427_alignment.py
```

### Dataset, analysis and first models (Parts I–IX)

```bash
python scripts/build_dataset.py --annotations official                      # filtered → windows_features_official.parquet
python scripts/build_dataset.py --annotations official --preprocess none \
    --skip-conversion                                                        # unfiltered → windows_features_official_raw.parquet
python scripts/plot_filtering.py                                             # one seizure before/after filtering
python scripts/analyze_features.py --annotations official --preprocess none  # unfiltered reference first
python scripts/analyze_features.py --annotations official                    # filtered report + comparison
python scripts/train_models.py                                               # docs/PARTS_VIII_IX.md
```

`--skip-conversion` reuses the labeled per-recording Parquet files after checking their labels against the current annotations. The provisional dataset (inferred offsets) is still built with `python scripts/build_dataset.py --annotations provisional`.

### Detection and the 12 models (Parts X–XI)

```bash
python scripts/predict_oof.py          # Part X: out-of-fold Random Forest predictions, models/rf_final.joblib
python scripts/build_augmented.py      # 8,230 extra seizure windows for the augmented mode, in their own file
python scripts/train_registry.py       # 12 models = DT, RF, KNN, SVM × unbalanced, balanced, augmented (about 1 hour)
python scripts/prepare_app_data.py     # filtered signals, reference channels, metadata, technician events, comparison curves
python scripts/evaluate_detection.py   # Part X–XI: threshold, outputs/detected_intervals.csv, figures, docs/PARTS_X_XI.md
python scripts/compare_models.py       # docs/MODEL_COMPARISON.md
```

`train_registry.py` writes:
- `outputs/predictions/<algo>_<mode>.parquet`: out-of-fold probabilities;
- `models/<algo>_<mode>.joblib`: final models trained on all 21 recordings;
- `models/registry.json`: features, best threshold, scores and training class distribution of every model, and the default model (best event F1).

`--modes augmented` (or any list of modes) retrains only those modes and keeps the other models of the registry.

### Detection from the terminal

```bash
python scripts/detect.py --recording 190304A_E [--algo dt|rf|knn|svm] [--mode unbalanced|balanced|augmented] [--threshold 0.5]
```

Without `--algo` and `--mode`, the default model is used.

## The app

The app only reads precomputed files. Run `build_dataset.py --annotations official`, `train_registry.py` and `prepare_app_data.py` once, then:

```bash
python app.py
```

Open <http://127.0.0.1:8050>. Nothing is trained at startup. If a precomputed file is missing, the app says which command to run.

The app opens in the light theme; the moon/sun button switches to the dark theme. The tabs in the header choose the page.

- **Home**: what the app is for, a real seizure with the best model's detection, the seven steps from raw EEG to detected seizures, the best model of each training mode, and buttons to the three parts. All numbers are read from the data and the registry.
- **Visualisation**: the data only. Choose a patient in the sidebar (sex, age, duration, number of seizures). The 19 channels are toggle chips grouped by scalp region ("Show all" brings them back), stacked in 10-20 order or overlaid. Each curve turns green during a real seizure, a diamond marks each seizure start, and the bottom row is "yes" during a seizure.
- **Prediction**: the model chosen in the header (algorithm: DT / RF / KNN / SVM; training mode: Unbalanced / Balanced / Augmented). A summary line gives the verdict for this patient (found, missed, false alarms) and the model's scores on all patients. The plot adds the model's probability, its detections and the threshold. The threshold slider starts at the model's best threshold and everything updates live.
- **Comparison**, read from top to bottom:
  - *Who wins*: the best of the compared models, then the others in rank order.
  - *Why*: event F1 against the threshold, precision-recall curves, training class distribution.
  - *Detail*: sortable ranking, confusion matrices, a timeline of the selected patient and event F1 per patient.

  Presets: "Same training mode", "Same algorithm, all modes", "All models", or a custom list. One colour per algorithm; solid = unbalanced, dashed or hatched = balanced, dotted = augmented.

Plot toolbar: previous/next page (also the ← → keys), previous/next seizure, **Spectrum** (frequency content of a 2 s window, also opened by clicking a curve), **Intervals** (real seizures and detections), "Save view as PNG" and "Download intervals CSV".

Sidebar view controls: stacked / overlay, page length (10 s, 30 s, 60 s, full), gain, technician notes, and reference channels (ECG, EMG, SLI in gray, never used by the models).

`?theme=dark&page=prediction&recording=211104B_D` in the URL opens a given state directly (`page` = home, visualisation, prediction or comparison).

Fonts (Google Fonts) and icons (Iconify) are loaded from the internet; offline, the app falls back to system fonts and draws no icons. Screenshots are in `outputs/figures/app/` (`python scripts/screenshot_app.py`); `outputs/figures/app/before/` keeps the previous design, described in [`docs/design/REDESIGN_PROPOSAL.md`](docs/design/REDESIGN_PROPOSAL.md).

## Reports

| Report | Content |
|---|---|
| [`docs/DATASET_PARTS_I_VI_OFFICIAL.md`](docs/DATASET_PARTS_I_VI_OFFICIAL.md) | Dataset with the official annotations: windows and class counts per recording |
| [`docs/PART_VII_ANALYSIS_OFFICIAL.md`](docs/PART_VII_ANALYSIS_OFFICIAL.md) | Class comparison and correlations of the features (filtered; `_RAW` = unfiltered) |
| [`docs/PARTS_VIII_IX.md`](docs/PARTS_VIII_IX.md) | First models, split by window and by recording |
| [`docs/PARTS_X_XI.md`](docs/PARTS_X_XI.md) | Temporal detection and error analysis (Random Forest) |
| [`docs/MODEL_COMPARISON.md`](docs/MODEL_COMPARISON.md) | The 12 models: scores, seeds, false alarms against hyperventilation and SLI, conclusion |
| [`docs/CHANNELS_8_VS_19.md`](docs/CHANNELS_8_VS_19.md) | 8 channels against 19 channels (the 8-channel results are in `outputs/archive_8ch/`) |
| [`docs/ANNOTATION_DECISIONS.md`](docs/ANNOTATION_DECISIONS.md) | Decisions on the annotation special cases |
| [`docs/DATASET_PARTS_I_VI.md`](docs/DATASET_PARTS_I_VI.md) | Provisional dataset (inferred offsets); `200625A_F` and `210427B_C` excluded there, see `data/interim/exclusions.csv` |

## Project structure

- `src/io.py`: MATLAB timetable and annotation loading, channel lists.
- `src/annotations.py`: provisional/official annotation resolution driven by `config/annotation_decisions.yaml`.
- `src/labels.py`: sample-level binary labels.
- `src/preprocessing.py`: band-pass filtering before segmentation.
- `src/segmentation.py`: 2 s windows with 50 % overlap.
- `src/features.py`: per-channel and across-channel time/frequency features.
- `src/detection.py`: window probabilities → seizure intervals, event-level comparison with annotations.
- `src/context.py`: hyperventilation and SLI periods, EMG bursts, for the false-alarm analysis.
- `src/app_data.py`, `src/app_figures.py`, `src/app_home.py`, `assets/app.css`: data access, themed figures, Home page and styles of `app.py`.
- `scripts/`: one script per step (see Commands).
- `outputs/`: figures, predictions, counts (`augmented_window_counts.csv`), app screenshots.
