"""Segment continuous EEG into 2 s windows with 50% overlap."""

from dataclasses import dataclass

import numpy as np

from src.io import FS


WINDOW_SECONDS = 2.0
OVERLAP = 0.5


@dataclass(frozen=True)
class Segmentation:
    starts: np.ndarray
    start_s: np.ndarray
    end_s: np.ndarray
    classes: np.ndarray
    window_samples: int
    hop_samples: int


def segment_labels(
    labels: np.ndarray,
    fs: float = FS,
    window_seconds: float = WINDOW_SECONDS,
    overlap: float = OVERLAP,
    positive_fraction: float = 0.5,
) -> Segmentation:
    window_samples = int(round(window_seconds * fs))
    hop_samples = int(round(window_samples * (1 - overlap)))
    if len(labels) < window_samples:
        empty = np.array([], dtype=np.int64)
        return Segmentation(empty, empty.astype(float), empty.astype(float), empty.astype(np.uint8), window_samples, hop_samples)
    starts = np.arange(0, len(labels) - window_samples + 1, hop_samples, dtype=np.int64)
    prefix = np.r_[0, np.cumsum(labels, dtype=np.int64)]
    positive_counts = prefix[starts + window_samples] - prefix[starts]
    classes = (positive_counts >= positive_fraction * window_samples).astype(np.uint8)
    return Segmentation(
        starts=starts,
        start_s=starts.astype(np.float64) / fs,
        end_s=(starts + window_samples).astype(np.float64) / fs,
        classes=classes,
        window_samples=window_samples,
        hop_samples=hop_samples,
    )


def window_view(signal: np.ndarray, segmentation: Segmentation) -> np.ndarray:
    """Return a read-only (windows, samples, channels) strided view."""
    if not len(segmentation.starts):
        return np.empty((0, segmentation.window_samples, signal.shape[1]), dtype=signal.dtype)
    view = np.lib.stride_tricks.sliding_window_view(signal, segmentation.window_samples, axis=0)
    # NumPy orders this view as (possible starts, channels, window samples).
    return view[segmentation.starts].transpose(0, 2, 1)

