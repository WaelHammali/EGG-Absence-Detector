"""Create the binary per-sample target from aligned seizure intervals."""

import numpy as np
import pandas as pd

from src.io import FS


def sample_labels(sample_count: int, intervals: pd.DataFrame, fs: float = FS) -> np.ndarray:
    """Label half-open seizure intervals [start_s, end_s) as uint8."""
    labels = np.zeros(sample_count, dtype=np.uint8)
    for row in intervals.itertuples(index=False):
        start = max(0, int(np.ceil(float(row.start_s) * fs - 1e-9)))
        end = min(sample_count, int(np.ceil(float(row.end_s) * fs - 1e-9)))
        if end > start:
            labels[start:end] = 1
    return labels


def add_class_column(frame: pd.DataFrame, intervals: pd.DataFrame, fs: float = FS) -> pd.DataFrame:
    result = frame.copy()
    result["Class"] = sample_labels(len(frame), intervals, fs)
    return result

