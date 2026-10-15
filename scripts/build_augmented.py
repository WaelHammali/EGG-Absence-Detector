"""Build the augmented seizure windows, kept apart from the official dataset.

Only class-1 windows are created, from the real filtered signal:
- shift: 2 s windows starting 0.25, 0.5 and 0.75 s after each regular window start (the regular
  grid has a 1 s hop), kept under the same rule as the dataset (at least 50% seizure samples);
- mirror: every seizure window, regular or shifted, with the left and right electrodes swapped
  (Fp1<->Fp2, F7<->F8, ...; Fz, Cz and Pz stay in place).

The official dataset is only read. Output: data/processed/datasets/augmented_shift_mirror/
windows_features.parquet and outputs/augmented_window_counts.csv. The windows are meant for training folds only.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import extract_features
from src.io import CHANNELS, FS
from src.paths import AUGMENTED, OFFICIAL, dataset_file, recordings_dir
from src.preprocessing import preprocess
from src.segmentation import WINDOW_SECONDS


DATASET = dataset_file(OFFICIAL)
OUTPUT = dataset_file(AUGMENTED)
COUNTS = ROOT / "outputs/augmented_window_counts.csv"
SHIFT_SECONDS = 0.25
MIRROR_PAIRS = [("Fp1", "Fp2"), ("F7", "F8"), ("F3", "F4"), ("T3", "T4"), ("C3", "C4"), ("T5", "T6"), ("P3", "P4"), ("O1", "O2")]


def mirror_order() -> list[int]:
    """Channel order after swapping the left and right electrodes."""
    swap = {left: right for left, right in MIRROR_PAIRS} | {right: left for left, right in MIRROR_PAIRS}
    return [CHANNELS.index(swap.get(channel, channel)) for channel in CHANNELS]


def seizure_windows(labels: np.ndarray, window: int, step: int) -> np.ndarray:
    """Starts of every window on the ``step`` grid with at least 50% seizure samples."""
    starts = np.arange(0, len(labels) - window + 1, step, dtype=np.int64)
    prefix = np.r_[0, np.cumsum(labels, dtype=np.int64)]
    return starts[prefix[starts + window] - prefix[starts] >= 0.5 * window]


def main() -> None:
    base = pd.read_parquet(DATASET)
    features = [column for column in base.columns if column not in {"recording", "subject", "window", "start_s", "end_s", "class"}]
    window = int(round(WINDOW_SECONDS * FS))
    step = int(round(SHIFT_SECONDS * FS))
    hop = window // 2
    order = mirror_order()
    parts, counts = [], []
    for recording, original in base.groupby("recording", sort=True):
        frame = pd.read_parquet(recordings_dir(OFFICIAL) / f"{recording}.parquet")
        signal = preprocess(frame.loc[:, CHANNELS].to_numpy(dtype=np.float32, copy=False), "bandpass")
        starts = seizure_windows(frame["Class"].to_numpy(dtype=np.uint8), window, step)
        regular = starts % hop == 0
        windows = np.stack([signal[start:start + window] for start in starts]) if len(starts) else np.empty((0, window, len(CHANNELS)), np.float32)
        plain = extract_features(windows)
        mirrored = extract_features(windows[:, :, order])

        # The windows of the regular grid must reproduce the official dataset exactly.
        official = original.loc[original["class"] == 1]
        assert np.array_equal(starts[regular], np.round(official["start_s"].to_numpy() * FS).astype(np.int64)), recording
        assert np.allclose(plain.loc[regular, features].to_numpy(), official[features].to_numpy(), rtol=1e-4, atol=1e-6), recording

        def rows(table: pd.DataFrame, keep: np.ndarray, kind: np.ndarray) -> pd.DataFrame:
            metadata = pd.DataFrame(
                {
                    "recording": recording,
                    "subject": recording.rsplit("_", 1)[1],
                    "start_s": (starts[keep] / FS).astype(np.float32),
                    "end_s": ((starts[keep] + window) / FS).astype(np.float32),
                    "augmentation": kind[keep],
                }
            )
            return pd.concat([metadata, table.loc[keep, features].reset_index(drop=True)], axis=1).assign(**{"class": np.uint8(1)})

        everything = np.ones(len(starts), dtype=bool)
        parts += [
            rows(plain, ~regular, np.full(len(starts), "shift")),
            rows(mirrored, everything, np.where(regular, "mirror", "shift+mirror")),
        ]
        counts.append(
            {
                "recording": recording, "official_class_1": int(regular.sum()), "shift": int((~regular).sum()),
                "mirror": int(regular.sum()), "shift+mirror": int((~regular).sum()),
            }
        )
        print(f"{recording}: official class 1 {regular.sum():,}, new windows {2 * len(starts) - regular.sum():,}", flush=True)

    result = pd.concat(parts, ignore_index=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(OUTPUT.stem + ".tmp.parquet")
    result.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(OUTPUT)

    table = pd.DataFrame(counts)
    table["augmented_total"] = table[["shift", "mirror", "shift+mirror"]].sum(axis=1)
    total = table.drop(columns="recording").sum().to_dict() | {"recording": "OVERALL"}
    table = pd.concat([table, pd.DataFrame([total])], ignore_index=True)
    table.to_csv(COUNTS, index=False)
    class_1 = int(total["official_class_1"] + total["augmented_total"])
    class_0 = int((base["class"] == 0).sum())
    print(f"Augmented windows: {len(result):,} ({result['augmentation'].value_counts().to_dict()})")
    print(f"Official + augmented: class 1 {class_1:,}, class 0 {class_0:,}, class-1 share {class_1 / (class_1 + class_0):.1%}")
    print(f"Dataset: {OUTPUT}\nCounts: {COUNTS}")


if __name__ == "__main__":
    main()
