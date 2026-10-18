# Data layout

Every recording set and every dataset has its own folder. Paths are defined once in [`src/paths.py`](../src/paths.py); scripts never build them by hand. `data/` is not in git: the commands in the README rebuild it from `data/raw/`.

```text
data/
├── raw/                                   original MAT files and Annotations.xlsx (read-only, never modified)
├── interim/                               aligned annotations and offsets
│   └── official/                          official annotations (annotations_clean.csv, offsets.csv, ...)
├── excel/
│   ├── official/                          labeled Excel copy of each recording (official annotations)
│   └── provisional_8ch/                   archive: first Excel conversion (8 channels)
└── processed/
    ├── recordings/                        one labeled signal per recording: Time, channels, Class
    │   ├── official/                      21 recordings, 19 EEG channels, official labels
    │   └── provisional_8ch/               archive, 20 recordings, 8 channels, provisional labels
    ├── datasets/                          one folder per dataset, each with windows_features.parquet + README.md
    │   ├── official/                      main dataset: filtered 0.5–40 Hz, 19 channels, 320 features
    │   ├── official_raw/                  same windows, unfiltered
    │   ├── official_8ch/                  archive: filtered, 8 channels, 144 features
    │   ├── provisional_8ch/               archive: first build (inferred offsets), 8 channels, unfiltered
    │   ├── augmented_shift_mirror/        extra seizure windows only (shifts + left-right mirror)
    │   ├── augmented_amplitude/           extra seizure windows only (random amplitude scaling)
    │   ├── augmented_noise/               extra seizure windows only (Gaussian noise)
    │   ├── augmented_time_mask/           extra seizure windows only (a random time span zeroed)
    │   ├── augmented_channel_dropout/     extra seizure windows only (1-3 channels zeroed); the best augmentation method
    │   └── augmented_combined/            extra seizure windows only (noise + channel dropout merged)
    └── app/                               precomputed files read by app.py (signals, references, curves)
```

## Datasets

| Folder | Windows | Seizure | Channels | Filter | Features | Built by | Used by |
|---|---:|---:|---:|---|---:|---|---|
| `official` | 31,419 | 1,244 (3.96 %) | 19 | 0.5–40 Hz | 320 | `build_dataset.py --annotations official` | all models, analysis, app |
| `official_raw` | 31,419 | 1,244 | 19 | none | 320 | `build_dataset.py --annotations official --preprocess none --skip-conversion` | Part VII comparison |
| `official_8ch` | 31,419 | 1,244 | 8 | 0.5–40 Hz | 144 | earlier version of the code | 8 vs 19 channels report |
| `provisional_8ch` | 30,174 | 3.82 % | 8 | none | 144 | earlier version of the code | Parts I–VI provisional report |
| `augmented_shift_mirror` | +8,230 (seizure only) | 100 % | 19 | 0.5–40 Hz | 320 | `build_augmented.py` | the `augmented` models |
| `augmented_amplitude` / `_noise` / `_time_mask` / `_channel_dropout` | +1,244 each (seizure only) | 100 % | 19 | 0.5–40 Hz | 320 | `build_augmented.py --method <name>` | the matching `augmented_<name>` models |
| `augmented_combined` | +2,488 (seizure only) | 100 % | 19 | 0.5–40 Hz | 320 | `build_augmented.py --method combined --include noise channel_dropout` | the `augmented_combined` models |

`official_8ch` and `provisional_8ch` are archives: no current script writes to them. A new provisional build (`build_dataset.py --annotations provisional`) goes to `recordings/provisional/` and `datasets/provisional/`, so the archives are never overwritten.

## Adding a new dataset

1. Choose a folder name and add it as a constant in `src/paths.py`.
2. Write `datasets/<name>/windows_features.parquet` with the same metadata columns (`recording`, `subject`, `start_s`, `end_s`, `class`) and a `README.md` saying what it contains and which script builds it.
3. Never write to another dataset's folder. Training-only data (augmentation, hard negatives) must only be added to training folds, never to test folds.

## Models and predictions

Models are named after their training mode: `models/<algo>_<mode>.joblib` and `outputs/predictions/<algo>_<mode>.parquet`, listed in `models/registry.json`, with `<algo>` = dt, rf, knn, svm and `<mode>` = unbalanced, balanced, augmented, augmented_amplitude, augmented_noise, augmented_time_mask, augmented_channel_dropout, or augmented_combined. All modes are tested on the `official` dataset; only the training folds differ.
