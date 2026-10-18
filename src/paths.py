"""Where every data set lives. All scripts take their paths from here.

data/processed/
    recordings/<set>/<recording>.parquet      labeled signal of each recording (Time, channels, Class)
    datasets/<name>/windows_features.parquet  one folder per window dataset
    app/                                      precomputed files read by app.py

See docs/DATA_LAYOUT.md for what each recording set and dataset contains.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
EXCEL = DATA / "excel"
PROCESSED = DATA / "processed"
RECORDINGS = PROCESSED / "recordings"
DATASETS = PROCESSED / "datasets"
APP_DATA = PROCESSED / "app"

# Dataset names (folders under data/processed/datasets/).
OFFICIAL = "official"  # official annotations, band-pass 0.5-40 Hz, 19 channels: the main dataset
OFFICIAL_RAW = "official_raw"  # same windows without the band-pass filter
OFFICIAL_8CH = "official_8ch"  # archive: official annotations, filtered, the original 8 channels
PROVISIONAL_8CH = "provisional_8ch"  # archive: first build, inferred offsets, 8 channels, unfiltered
AUGMENTED = "augmented_shift_mirror"  # extra seizure windows only (0.25 s shifts + left-right mirror)
AUGMENTED_AMPLITUDE = "augmented_amplitude"  # extra seizure windows only (random 0.6-1.6x amplitude scaling)
AUGMENTED_NOISE = "augmented_noise"  # extra seizure windows only (noise scaled to each window's own std)
AUGMENTED_TIME_MASK = "augmented_time_mask"  # extra seizure windows only (a random time span zeroed out)
AUGMENTED_CHANNEL_DROPOUT = "augmented_channel_dropout"  # extra seizure windows only (1-3 random channels zeroed out)
AUGMENTED_COMBINED = "augmented_combined"  # concatenation of whichever augmentation methods helped individually

# method name -> dataset name, the shared source of truth for build_augmented.py and train_registry.py
AUGMENTED_METHODS = {
    "shift_mirror": AUGMENTED,
    "amplitude": AUGMENTED_AMPLITUDE,
    "noise": AUGMENTED_NOISE,
    "time_mask": AUGMENTED_TIME_MASK,
    "channel_dropout": AUGMENTED_CHANNEL_DROPOUT,
    "combined": AUGMENTED_COMBINED,
}


def recordings_dir(annotation_set: str) -> Path:
    """Labeled per-recording signals of one annotation set (official, provisional, provisional_8ch)."""
    return RECORDINGS / annotation_set


def dataset_dir(name: str) -> Path:
    return DATASETS / name


def dataset_file(name: str) -> Path:
    return DATASETS / name / "windows_features.parquet"
