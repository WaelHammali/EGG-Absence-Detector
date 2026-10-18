"""Build extra, synthetic seizure windows for the "augmented" classical training modes.

Only class-1 windows are created, from the real filtered signal of each recording -- the caller
(scripts/train_registry.py) restricts them to training folds only. --method selects which
dataset to build; default is the original method.

- shift_mirror (default): 2 s windows starting 0.25, 0.5 and 0.75 s after each regular window
  start (the regular grid has a 1 s hop), kept under the same rule as the official dataset (at
  least 50% seizure samples), each one also mirrored left-right (Fp1<->Fp2, F7<->F8, ...; Fz, Cz
  and Pz stay in place).
- amplitude: every real seizure window, rescaled by a random 0.6-1.6x gain.
- noise: every real seizure window, with Gaussian noise added, scaled to that window's own
  standard deviation.
- time_mask: every real seizure window, with a random contiguous time span (up to 25% of the
  window) zeroed out.
- channel_dropout: every real seizure window, with 1 to 3 random channels zeroed out.
- combined: concatenates the already-built parquets of --include methods -- no recomputation.

The official dataset is only read. Windows are meant for training folds only.

Usage:
    python scripts/build_augmented.py                              # shift_mirror (default)
    python scripts/build_augmented.py --method noise
    python scripts/build_augmented.py --method combined --include shift_mirror noise
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augment_classical import add_noise, dropout_channels, mask_time, scale_amplitude
from src.features import extract_features
from src.io import CHANNELS, FS
from src.paths import AUGMENTED_METHODS, OFFICIAL, dataset_file, recordings_dir
from src.preprocessing import preprocess
from src.segmentation import WINDOW_SECONDS


DATASET = dataset_file(OFFICIAL)
SHIFT_SECONDS = 0.25
SEED = 0  # simple methods only (amplitude/noise/time_mask/channel_dropout); shift_mirror has no randomness
MIRROR_PAIRS = [("Fp1", "Fp2"), ("F7", "F8"), ("F3", "F4"), ("T3", "T4"), ("C3", "C4"), ("T5", "T6"), ("P3", "P4"), ("O1", "O2")]
SIMPLE_TRANSFORMS = {
    "amplitude": scale_amplitude,
    "noise": add_noise,
    "time_mask": mask_time,
    "channel_dropout": dropout_channels,
}
METHOD_DESCRIPTIONS = {
    "amplitude": "rescaled by a random 0.6-1.6x gain (src/augment_classical.py:scale_amplitude)",
    "noise": "with Gaussian noise added, scaled to that window's own standard deviation (src/augment_classical.py:add_noise)",
    "time_mask": "with a random contiguous time span, up to 25% of the window, zeroed out (src/augment_classical.py:mask_time)",
    "channel_dropout": "with 1 to 3 random channels zeroed out (src/augment_classical.py:dropout_channels)",
}


def output_paths(method: str) -> tuple[Path, Path | None]:
    """(dataset output, counts CSV). shift_mirror keeps its original counts filename unchanged
    (scripts/compare_models.py reads it); combined has no per-recording counts to report."""
    output = dataset_file(AUGMENTED_METHODS[method])
    if method == "shift_mirror":
        return output, ROOT / "outputs/augmented_window_counts.csv"
    if method == "combined":
        return output, None
    return output, ROOT / f"outputs/augmented_window_counts_{method}.csv"


def mirror_order() -> list[int]:
    """Channel order after swapping the left and right electrodes."""
    swap = {left: right for left, right in MIRROR_PAIRS} | {right: left for left, right in MIRROR_PAIRS}
    return [CHANNELS.index(swap.get(channel, channel)) for channel in CHANNELS]


def seizure_windows(labels: np.ndarray, window: int, step: int) -> np.ndarray:
    """Starts of every window on the ``step`` grid with at least 50% seizure samples."""
    starts = np.arange(0, len(labels) - window + 1, step, dtype=np.int64)
    prefix = np.r_[0, np.cumsum(labels, dtype=np.int64)]
    return starts[prefix[starts + window] - prefix[starts] >= 0.5 * window]


def build_shift_mirror(base: pd.DataFrame, features: list[str], window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The original method, unchanged: shifted and mirrored seizure windows."""
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
    table = pd.DataFrame(counts)
    table["augmented_total"] = table[["shift", "mirror", "shift+mirror"]].sum(axis=1)
    total = table.drop(columns="recording").sum().to_dict() | {"recording": "OVERALL"}
    table = pd.concat([table, pd.DataFrame([total])], ignore_index=True)
    return result, table


def build_simple(method: str, base: pd.DataFrame, features: list[str], window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One new synthetic window per real seizure window: the official window, transformed once."""
    transform = SIMPLE_TRANSFORMS[method]
    rng = np.random.default_rng(SEED)
    parts, counts = [], []
    for recording, original in base.groupby("recording", sort=True):
        frame = pd.read_parquet(recordings_dir(OFFICIAL) / f"{recording}.parquet")
        signal = preprocess(frame.loc[:, CHANNELS].to_numpy(dtype=np.float32, copy=False), "bandpass")
        official = original.loc[original["class"] == 1]
        starts = np.round(official["start_s"].to_numpy() * FS).astype(np.int64)
        windows = np.stack([signal[start:start + window] for start in starts]) if len(starts) else np.empty((0, window, len(CHANNELS)), np.float32)

        # The untransformed windows must reproduce the official dataset exactly (same check as shift_mirror).
        plain = extract_features(windows)
        assert np.allclose(plain[features].to_numpy(), official[features].to_numpy(), rtol=1e-4, atol=1e-6), recording

        transformed = extract_features(transform(windows, rng))
        metadata = pd.DataFrame(
            {
                "recording": recording,
                "subject": recording.rsplit("_", 1)[1],
                "start_s": official["start_s"].to_numpy(),
                "end_s": official["end_s"].to_numpy(),
                "augmentation": method,
            }
        )
        parts.append(pd.concat([metadata, transformed[features]], axis=1).assign(**{"class": np.uint8(1)}))
        counts.append({"recording": recording, "official_class_1": len(starts), method: len(starts)})
        print(f"{recording}: official class 1 {len(starts):,}, new windows {len(starts):,}", flush=True)

    result = pd.concat(parts, ignore_index=True)
    table = pd.DataFrame(counts)
    total = table.drop(columns="recording").sum().to_dict() | {"recording": "OVERALL"}
    table = pd.concat([table, pd.DataFrame([total])], ignore_index=True)
    return result, table


def build_combined(include: list[str]) -> pd.DataFrame:
    """Concatenate the already-built parquets of ``include`` methods -- no recomputation."""
    parts = []
    for method in include:
        path = dataset_file(AUGMENTED_METHODS[method])
        if not path.exists():
            raise SystemExit(f"{path.relative_to(ROOT)} not found: run python scripts/build_augmented.py --method {method}")
        parts.append(pd.read_parquet(path))
    return pd.concat(parts, ignore_index=True)


def write_readme(output: Path, method: str, include: list[str] | None, n_windows: int) -> None:
    if method == "shift_mirror":
        text = (
            f"# augmented_shift_mirror\n\nExtra seizure windows only ({n_windows:,}), cut from the filtered signal of "
            "`official`: windows shifted by 0.25, 0.5 and 0.75 s inside seizures, and the left-right mirror of every "
            "seizure window. Added to the training folds of the `augmented` models only.\n"
            "Built by: python scripts/build_augmented.py (counts: outputs/augmented_window_counts.csv)\n"
        )
    elif method == "combined":
        text = (
            f"# augmented_combined\n\nConcatenation of already-built augmented datasets ({n_windows:,} windows total, "
            f"from: {', '.join(include)}) -- see the `augmentation` column for which method produced each row. Added "
            "to the training folds of the `augmented_combined` model only.\n"
            "Built by: python scripts/build_augmented.py --method combined --include <method> [<method> ...]\n"
        )
    else:
        text = (
            f"# {AUGMENTED_METHODS[method]}\n\nExtra seizure windows only ({n_windows:,}), one per real seizure window "
            f"of `official`, {METHOD_DESCRIPTIONS[method]}. Added to the training folds of the `augmented_{method}` "
            f"model only.\nBuilt by: python scripts/build_augmented.py --method {method} "
            f"(counts: outputs/augmented_window_counts_{method}.csv)\n"
        )
    (output.parent / "README.md").write_text(text)


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=list(AUGMENTED_METHODS), default="shift_mirror")
    parser.add_argument("--include", nargs="+", choices=[m for m in AUGMENTED_METHODS if m != "combined"], help="Methods to concatenate (--method combined only)")
    args = parser.parse_args()
    if args.method == "combined" and not args.include:
        raise SystemExit("--method combined needs --include <method> [<method> ...]")

    output, counts_path = output_paths(args.method)
    base = pd.read_parquet(DATASET)
    features = [column for column in base.columns if column not in {"recording", "subject", "window", "start_s", "end_s", "class"}]
    window = int(round(WINDOW_SECONDS * FS))

    if args.method == "shift_mirror":
        result, table = build_shift_mirror(base, features, window)
    elif args.method == "combined":
        result, table = build_combined(args.include), None
    else:
        result, table = build_simple(args.method, base, features, window)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.parquet")
    result.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output)
    write_readme(output, args.method, args.include, len(result))

    if table is not None:
        table.to_csv(counts_path, index=False)
        print(f"Counts: {counts_path}")
    class_1 = int((base["class"] == 1).sum() + len(result))
    class_0 = int((base["class"] == 0).sum())
    print(f"Augmented windows: {len(result):,} ({result['augmentation'].value_counts().to_dict()})")
    print(f"Official + augmented: class 1 {class_1:,}, class 0 {class_0:,}, class-1 share {class_1 / (class_1 + class_0):.1%}")
    print(f"Dataset: {output}")


if __name__ == "__main__":
    main()
