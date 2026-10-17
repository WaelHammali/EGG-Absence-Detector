"""Package the cleaned EEG for training the deep learning models (scripts/train_cnn.py).

Writes data/cnn/ (git-ignored, about 800 MB), to copy to the GPU machine that trains them:
- signals/<recording>.parquet: the 19 EEG channels after the 0.5-40 Hz band-pass (float32, µV)
  and the per-sample label `label` (official annotations);
- windows.csv: the 31,419 windows of the official dataset (recording, window, start_s, end_s,
  true_class) with the outer `fold` (1-5) used by every model here, so the deep models are scored
  on exactly the same windows and folds;
- annotations.csv, recordings.csv (duration, sex, age).

Only recording codes are exported: no names or dates. Keep the package private: it is patient EEG.
"""

from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io import CHANNELS
from src.paths import APP_DATA, DATA, OFFICIAL, recordings_dir


OUT = DATA / "cnn"
PREDICTIONS = ROOT / "outputs/predictions/svm_unbalanced.parquet"
ANNOTATIONS = ROOT / "data/interim/official/annotations_clean.csv"


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "signals").mkdir(parents=True)
    windows = pd.read_parquet(PREDICTIONS, columns=["recording", "window", "start_s", "end_s", "true_class", "fold"])
    metadata = pd.read_csv(APP_DATA / "recording_metadata.csv")
    for recording in sorted(windows["recording"].unique()):
        signal = pd.read_parquet(APP_DATA / f"signals/{recording}.parquet", columns=list(CHANNELS))
        labels = pd.read_parquet(recordings_dir(OFFICIAL) / f"{recording}.parquet", columns=["Class"])["Class"].to_numpy(np.uint8)
        if len(signal) != len(labels):
            raise ValueError(f"{recording}: {len(signal)} filtered samples but {len(labels)} labels")
        signal["label"] = labels
        signal.to_parquet(OUT / f"signals/{recording}.parquet", index=False, compression="zstd")
        print(f"{recording}: {len(signal):,} samples, {int(labels.sum()):,} seizure samples", flush=True)
    windows.to_csv(OUT / "windows.csv", index=False)
    pd.read_csv(ANNOTATIONS)[["recording", "start_s", "end_s"]].to_csv(OUT / "annotations.csv", index=False)
    metadata[["recording", "duration_s", "sex", "age"]].to_csv(OUT / "recordings.csv", index=False)
    size = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()) / 1e6
    print(f"{len(windows):,} windows, {windows['true_class'].sum():,} seizure windows, folds {sorted(windows['fold'].unique())}")
    print(f"Package: {OUT} ({size:.0f} MB). Train with: python scripts/train_cnn.py --data data/cnn")


if __name__ == "__main__":
    main()
