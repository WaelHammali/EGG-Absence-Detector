"""Read-only access to the precomputed files used by app.py. No training happens here."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.detection import detect_all, event_metrics, match_events, windows_to_intervals
from src.io import CHANNELS, FS


ROOT = Path(__file__).resolve().parents[1]
APP_DATA = ROOT / "data/processed/app"
REQUIRED = {
    "outputs/predictions_oof_models.parquet": "python scripts/predict_oof.py",
    "outputs/model_comparison.csv": "python scripts/predict_oof.py",
    "data/processed/app/recording_metadata.csv": "python scripts/prepare_app_data.py",
    "data/processed/app/technician_events.csv": "python scripts/prepare_app_data.py",
    "data/interim/official/annotations_clean.csv": "python scripts/build_dataset.py --annotations official",
}


class MissingData(Exception):
    """Raised with a user-facing message when a precomputed file is absent."""


DEFAULT_MODEL = "Random Forest"


@dataclass(frozen=True)
class AppData:
    predictions: dict[str, pd.DataFrame]  # model name -> out-of-fold window probabilities
    comparison: pd.DataFrame  # one row per model: best threshold and overall scores
    annotations: pd.DataFrame
    metadata: pd.DataFrame
    events: pd.DataFrame

    @property
    def recordings(self) -> list[str]:
        return self.metadata["recording"].tolist()

    @property
    def models(self) -> list[str]:
        return self.comparison["model"].tolist()

    def best_threshold(self, model: str) -> float:
        """Threshold maximizing event-level F1 for this model on out-of-fold predictions."""
        return float(self.comparison.set_index("model").loc[model, "threshold"])

    def info(self, recording: str) -> pd.Series:
        return self.metadata.set_index("recording").loc[recording]

    def real(self, recording: str) -> pd.DataFrame:
        frame = self.annotations.loc[self.annotations["recording"] == recording, ["start_s", "end_s"]]
        return frame.sort_values("start_s").reset_index(drop=True)

    def windows(self, recording: str, model: str = DEFAULT_MODEL) -> pd.DataFrame:
        frame = self.predictions[model]
        return frame.loc[frame["recording"] == recording]

    def technician(self, recording: str) -> pd.DataFrame:
        return self.events.loc[self.events["recording"] == recording]

    def detect(self, recording: str, threshold: float, model: str = DEFAULT_MODEL) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Detected intervals, event matches and metrics for one recording and one model."""
        group = self.windows(recording, model)
        detected = windows_to_intervals(group["start_s"], group["end_s"], group["proba"], threshold)
        matches = match_events(detected, self.real(recording))
        metrics = event_metrics(matches, len(detected))
        metrics["detections"] = len(detected)
        predicted = group["proba"].to_numpy() >= threshold
        truth = group["true_class"].to_numpy() == 1
        tp, fp, fn = int((predicted & truth).sum()), int((predicted & ~truth).sum()), int((~predicted & truth).sum())
        metrics["window"] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": int((~predicted & ~truth).sum()),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
        }
        return detected, matches, metrics

    def detect_everything(self, threshold: float, model: str = DEFAULT_MODEL) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        detected, matches = detect_all(self.predictions[model], self.annotations, threshold)
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
    table = pd.read_parquet(ROOT / "outputs/predictions_oof_models.parquet")
    predictions = {model: group.drop(columns="model").reset_index(drop=True) for model, group in table.groupby("model", sort=False)}
    comparison = pd.read_csv(ROOT / "outputs/model_comparison.csv")
    annotations = pd.read_csv(ROOT / "data/interim/official/annotations_clean.csv")
    metadata = pd.read_csv(APP_DATA / "recording_metadata.csv").fillna({"absence_type": ""})
    events = pd.read_csv(APP_DATA / "technician_events.csv")
    return AppData(predictions, comparison, annotations, metadata, events)


@lru_cache(maxsize=4)
def signal(recording: str) -> pd.DataFrame:
    """Filtered signal of one recording: Time + the EEG channels (float32)."""
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
