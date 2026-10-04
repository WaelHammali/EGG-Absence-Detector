"""Build one window-feature row per 2 s / 50%-overlap EEG segment."""

from pathlib import Path
import csv
import json
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import extract_features
from src.io import CHANNELS, annotations_for_recording, load_annotations
from src.segmentation import segment_labels, window_view


PROCESSED = ROOT / "data/processed"
ANNOTATIONS = ROOT / "data/interim/annotations_clean.csv"
EXCLUSIONS = ROOT / "data/interim/exclusions.csv"
OUTPUT = PROCESSED / "windows_features.parquet"
REPORT_CSV = ROOT / "outputs/window_counts.csv"
REPORT_JSON = ROOT / "outputs/dataset_summary.json"
REPORT_MD = ROOT / "docs/DATASET_PARTS_I_VI.md"


def seizure_window_coverage(segmentation, intervals: pd.DataFrame, fs: float = 256.0) -> tuple[int, int]:
    positive_starts = segmentation.starts[segmentation.classes == 1]
    positive_ends = positive_starts + segmentation.window_samples
    represented = 0
    for interval in intervals.itertuples(index=False):
        seizure_start = int(np.ceil(float(interval.start_s) * fs - 1e-9))
        seizure_end = int(np.ceil(float(interval.end_s) * fs - 1e-9))
        overlap = np.minimum(positive_ends, seizure_end) - np.maximum(positive_starts, seizure_start)
        if np.any(overlap >= segmentation.window_samples // 2):
            represented += 1
    return represented, len(intervals)


def main() -> None:
    annotations = load_annotations(ANNOTATIONS)
    excluded = set(pd.read_csv(EXCLUSIONS)["recording"])
    recording_files = sorted(
        path for path in PROCESSED.glob("*.parquet")
        if path.stem != "windows_features" and path.stem not in excluded
    )
    expected = sorted(set(annotations["recording"]) - excluded)
    actual = [path.stem for path in recording_files]
    if actual != expected:
        raise ValueError(f"Converted recordings do not match expected set. Expected {expected}; found {actual}")

    temporary = OUTPUT.with_name("windows_features.tmp.parquet")
    writer = None
    report_rows = []
    try:
        for path in recording_files:
            recording = path.stem
            frame = pd.read_parquet(path, engine="pyarrow")
            required = ["Time", *CHANNELS, "Class"]
            if frame.columns.tolist() != required:
                raise ValueError(f"{recording}: unexpected columns {frame.columns.tolist()}")
            signal = frame.loc[:, CHANNELS].to_numpy(dtype=np.float32, copy=False)
            labels = frame["Class"].to_numpy(dtype=np.uint8, copy=False)
            segmentation = segment_labels(labels)
            windows = window_view(signal, segmentation)
            features = extract_features(windows)
            metadata = pd.DataFrame(
                {
                    "recording": recording,
                    "subject": recording.rsplit("_", 1)[1],
                    "window": np.arange(len(segmentation.starts), dtype=np.int32),
                    "start_s": segmentation.start_s.astype(np.float32),
                    "end_s": segmentation.end_s.astype(np.float32),
                }
            )
            output = pd.concat([metadata, features], axis=1)
            output["class"] = segmentation.classes
            table = pa.Table.from_pandas(output, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)

            intervals = annotations_for_recording(annotations, recording)
            represented, seizure_count = seizure_window_coverage(segmentation, intervals)
            class_1 = int(segmentation.classes.sum())
            total = len(segmentation.classes)
            report_rows.append(
                {
                    "recording": recording,
                    "total_windows": total,
                    "class_0": total - class_1,
                    "class_1": class_1,
                    "class_0_percent": 100 * (total - class_1) / total,
                    "class_1_percent": 100 * class_1 / total,
                    "seizures_total": seizure_count,
                    "seizures_with_positive_window": represented,
                    "seizures_without_positive_window": seizure_count - represented,
                }
            )
            print(
                f"{recording}: windows={total:,}, class1={class_1:,}, "
                f"represented seizures={represented}/{seizure_count}",
                flush=True,
            )
            del frame, signal, labels, windows, features, output, table
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("No recordings were written")
    temporary.replace(OUTPUT)

    total_windows = sum(row["total_windows"] for row in report_rows)
    total_1 = sum(row["class_1"] for row in report_rows)
    total_seizures = sum(row["seizures_total"] for row in report_rows)
    represented = sum(row["seizures_with_positive_window"] for row in report_rows)
    overall = {
        "recording": "OVERALL",
        "total_windows": total_windows,
        "class_0": total_windows - total_1,
        "class_1": total_1,
        "class_0_percent": 100 * (total_windows - total_1) / total_windows,
        "class_1_percent": 100 * total_1 / total_windows,
        "seizures_total": total_seizures,
        "seizures_with_positive_window": represented,
        "seizures_without_positive_window": total_seizures - represented,
    }
    all_rows = report_rows + [overall]
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_CSV.open("w", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(overall))
        writer_csv.writeheader()
        writer_csv.writerows(all_rows)
    summary = {
        "parameters": {
            "sampling_frequency_hz": 256,
            "window_seconds": 2,
            "window_samples": 512,
            "overlap_percent": 50,
            "hop_samples": 256,
            "positive_rule": "at least 50% (256/512 samples) labeled seizure",
            "channels": list(CHANNELS),
            "excluded_recordings": sorted(excluded),
        },
        "per_recording": report_rows,
        "overall": overall,
        "dataset_columns": pq.read_schema(OUTPUT).names,
    }
    REPORT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(summary)
    print(f"Dataset: {OUTPUT}")
    print(f"Overall: {overall}")


def write_markdown(summary: dict) -> None:
    rows = summary["per_recording"]
    overall = summary["overall"]
    lines = [
        "# Dataset report — Parts I–VI",
        "",
        "The dataset contains 2 s windows (512 samples at 256 Hz) with a 1 s hop. A window is positive when at least 256 samples are inside a cleaned seizure interval. Features use only Fp1, Fp2, C3, C4, T3, T4, O1 and O2.",
        "",
        "`200625A_F` is excluded because it is unannotated. `210427B_C` is excluded because both competing alignments contain convincing spike-and-wave activity, so its single Excel interval cannot be assigned uniquely. The comparison is [here](../outputs/figures/offset_alignment/210427B_C_candidate_comparison.png).",
        "",
        "| Recording | Windows | Class 0 | Class 1 | Class 0 % | Class 1 % | Seizures represented / total |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows + [overall]:
        lines.append(
            f"| {row['recording']} | {row['total_windows']} | {row['class_0']} | {row['class_1']} | "
            f"{row['class_0_percent']:.3f} | {row['class_1_percent']:.3f} | "
            f"{row['seizures_with_positive_window']} / {row['seizures_total']} |"
        )
    lines += [
        "",
        "Feature columns comprise, per channel and averaged across channels: mean, standard deviation, variance, minimum, maximum, amplitude, RMS, energy, absolute Delta/Theta/Alpha/Beta power, and relative Delta/Theta/Alpha/Beta power. Spectral power uses a mean-removed 2 s signal, Hann window and one-sided rFFT periodogram.",
        "",
        "Artifacts:",
        "",
        "- `data/excel/<recording>.xlsx`: labeled samples, split automatically at 1,048,575 data rows per sheet.",
        "- `data/processed/<recording>.parquet`: fast labeled sample copies.",
        "- `data/processed/windows_features.parquet`: final Parts I–VI feature table.",
        "- `outputs/window_counts.csv` and `outputs/dataset_summary.json`: machine-readable counts and schema.",
        "",
        "No model training or train/test splitting has been performed.",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
