"""Read-only access to the precomputed files used by app.py. No training happens here."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.detection import detect_all, event_metrics, match_events, windows_to_intervals
from src.io import CHANNELS, FS


ROOT = Path(__file__).resolve().parents[1]
APP_DATA = ROOT / "data/processed/app"
REQUIRED = {
    "outputs/predictions_oof.parquet": "python scripts/predict_oof.py",
    "data/processed/app/recording_metadata.csv": "python scripts/prepare_app_data.py",
    "data/processed/app/technician_events.csv": "python scripts/prepare_app_data.py",
    "data/interim/official/annotations_clean.csv": "python scripts/build_dataset.py --annotations official",
}


class MissingData(Exception):
    """Raised with a user-facing message when a precomputed file is absent."""


@dataclass(frozen=True)
class AppData:
    predictions: pd.DataFrame
    annotations: pd.DataFrame
    metadata: pd.DataFrame
    events: pd.DataFrame
    default_threshold: float

    @property
    def recordings(self) -> list[str]:
        return self.metadata["recording"].tolist()

    def info(self, recording: str) -> pd.Series:
        return self.metadata.set_index("recording").loc[recording]

    def real(self, recording: str) -> pd.DataFrame:
        frame = self.annotations.loc[self.annotations["recording"] == recording, ["start_s", "end_s"]]
        return frame.sort_values("start_s").reset_index(drop=True)

    def windows(self, recording: str) -> pd.DataFrame:
        return self.predictions.loc[self.predictions["recording"] == recording]

    def technician(self, recording: str) -> pd.DataFrame:
        return self.events.loc[self.events["recording"] == recording]

    def detect(self, recording: str, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Detected intervals, event matches and event metrics for one recording."""
        group = self.windows(recording)
        detected = windows_to_intervals(group["start_s"], group["end_s"], group["proba"], threshold)
        matches = match_events(detected, self.real(recording))
        return detected, matches, event_metrics(matches, len(detected))

    def detect_everything(self, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        detected, matches = detect_all(self.predictions, self.annotations, threshold)
        metrics = event_metrics(matches, len(detected))
        metrics["detections"] = len(detected)
        metrics["false_alarms_per_hour"] = metrics["fp"] / (self.metadata["duration_s"].sum() / 3600)
        return detected, matches, metrics


def load() -> AppData:
    missing = [(path, command) for path, command in REQUIRED.items() if not (ROOT / path).exists()]
    if missing:
        steps = list(dict.fromkeys(command for _, command in missing))
        raise MissingData(
            "Some precomputed files are missing: " + ", ".join(path for path, _ in missing)
            + ". Run: " + " ; ".join(steps)
        )
    predictions = pd.read_parquet(ROOT / "outputs/predictions_oof.parquet")
    annotations = pd.read_csv(ROOT / "data/interim/official/annotations_clean.csv")
    metadata = pd.read_csv(APP_DATA / "recording_metadata.csv").fillna({"absence_type": ""})
    events = pd.read_csv(APP_DATA / "technician_events.csv")
    summary = ROOT / "outputs/detection_summary.json"
    threshold = float(json.loads(summary.read_text())["threshold"]) if summary.exists() else 0.5
    return AppData(predictions, annotations, metadata, events, threshold)


@lru_cache(maxsize=4)
def signal(recording: str) -> pd.DataFrame:
    """Filtered signal of one recording: Time + 8 channels (float32)."""
    path = APP_DATA / f"signals/{recording}.parquet"
    if not path.exists():
        raise MissingData(f"Filtered signal for {recording} is missing. Run: python scripts/prepare_app_data.py")
    return pd.read_parquet(path, columns=["Time", *CHANNELS])


@lru_cache(maxsize=32)
def amplitude_scale(recording: str) -> float:
    """Robust full-scale amplitude shared by all channels, so sensitivity is uniform across rows."""
    values = signal(recording).loc[:, list(CHANNELS)].to_numpy()
    # High percentile so that seizure discharges fit in a row at gain ×1.
    return float(np.percentile(np.abs(values[::4]), 99.9))


BANDS = {"Delta": (0.5, 4.0), "Theta": (4.0, 8.0), "Alpha": (8.0, 13.0), "Beta": (13.0, 30.0)}


def spectrum(recording: str, center_s: float, channels: list[str], seconds: float = 2.0) -> dict:
    """Hann-windowed periodogram of the ``seconds``-long window centered on ``center_s``."""
    frame = signal(recording)
    samples = int(round(seconds * FS))
    first = int(round(center_s * FS)) - samples // 2
    first = max(0, min(first, len(frame) - samples))
    block = frame.iloc[first : first + samples]
    hann = np.hanning(samples)
    frequencies = np.fft.rfftfreq(samples, d=1 / FS)
    result = {"start_s": first / FS, "end_s": (first + samples) / FS, "frequencies": frequencies, "psd": {}, "relative": {}}
    total_mask = (frequencies >= 0.5) & (frequencies <= 30.0)
    for channel in channels:
        values = block[channel].to_numpy(dtype=np.float64)
        transformed = np.fft.rfft((values - values.mean()) * hann)
        psd = (transformed.real**2 + transformed.imag**2) / (FS * np.sum(hann * hann))
        psd[1:-1] *= 2.0
        result["psd"][channel] = psd
        total = max(psd[total_mask].sum(), 1e-20)
        result["relative"][channel] = {
            band: float(psd[(frequencies >= low) & ((frequencies < high) | ((band == "Beta") & (frequencies <= high)))].sum() / total)
            for band, (low, high) in BANDS.items()
        }
    return result
