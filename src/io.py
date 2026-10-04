"""Read MATLAB EEG recordings and the derived annotation table."""

from pathlib import Path

from matio import load_from_mat
import numpy as np
import pandas as pd


FS = 256.0
CHANNELS = ("Fp1", "Fp2", "C3", "C4", "T3", "T4", "O1", "O2")
RAW_CHANNELS = tuple(f"EEG{channel}" for channel in CHANNELS)


def recording_ids(raw_dir: str | Path) -> list[str]:
    """Return sorted recording IDs having a d-file."""
    raw_dir = Path(raw_dir)
    return sorted(path.name.removesuffix("_0000d.mat") for path in raw_dir.glob("*_0000d.mat"))


def load_recording(path: str | Path) -> pd.DataFrame:
    """Load one d-file as float32 [Time, Fp1, Fp2, C3, C4, T3, T4, O1, O2]."""
    path = Path(path)
    table = load_from_mat(path, variable_names=["output_d"])["output_d"]
    missing = [channel for channel in RAW_CHANNELS if channel not in table.columns]
    if missing:
        raise ValueError(f"{path.name}: missing required EEG channels: {missing}")
    arrays = []
    for raw_channel in RAW_CHANNELS:
        arrays.append(np.concatenate([np.asarray(block).reshape(-1) for block in table[raw_channel]]))
    sample_count = len(arrays[0])
    if any(len(values) != sample_count for values in arrays):
        raise ValueError(f"{path.name}: channel lengths differ")
    matrix = np.column_stack(arrays).astype(np.float32, copy=False)
    frame = pd.DataFrame(matrix, columns=CHANNELS)
    frame.insert(0, "Time", (np.arange(sample_count, dtype=np.float64) / FS).astype(np.float32))
    return frame


def load_annotations(path: str | Path) -> pd.DataFrame:
    """Load and validate the cleaned seizure intervals."""
    frame = pd.read_csv(path)
    required = {"recording", "seizure_id", "start_s", "end_s", "duration_s", "source_row", "note"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Annotation file lacks columns: {sorted(missing)}")
    if (frame["end_s"] <= frame["start_s"]).any():
        raise ValueError("Every seizure interval must have end_s > start_s")
    return frame


def annotations_for_recording(annotations: pd.DataFrame, recording: str) -> pd.DataFrame:
    return annotations.loc[annotations["recording"] == recording].sort_values("start_s").reset_index(drop=True)

