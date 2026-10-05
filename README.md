# eeg-absence-detector

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

Only Fp1, Fp2, C3, C4, T3, T4, O1 and O2 are used. EEG amplitudes and feature arrays are stored as `float32`. ECG, EMG and SLI are excluded.

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

The provisional counts and feature definition are documented in [`docs/DATASET_PARTS_I_VI.md`](docs/DATASET_PARTS_I_VI.md). `200625A_F` and `210427B_C` are excluded for the reasons recorded in `data/interim/exclusions.csv`.

## Project structure

- `src/io.py`: MATLAB timetable and annotation loading.
- `src/labels.py`: sample-level binary labels.
- `src/preprocessing.py`: band-pass filtering before segmentation.
- `src/annotations.py`: provisional/official annotation resolution driven by `config/annotation_decisions.yaml`.
- `src/segmentation.py`: 2 s windows with 50% overlap.
- `src/features.py`: per-channel and across-channel time/frequency features.
- `scripts/convert_to_excel.py`: labeled Excel and Parquet conversion.
- `scripts/build_dataset.py`: consolidated window feature table and count report.
- `outputs/figures/offset_alignment/`: offset searches and seizure-centered waveform checks.
