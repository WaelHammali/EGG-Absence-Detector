"""Resolve annotations, convert recordings, and build the window-feature dataset."""

from argparse import ArgumentParser
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

from convert_to_excel import convert_recording
from src.annotations import resolve_annotations
from src.features import extract_features
from src.io import CHANNELS, annotations_for_recording
from src.labels import sample_labels
from src.preprocessing import PREPROCESSING, describe, preprocess
from src.segmentation import segment_labels, window_view


PROCESSED = ROOT / "data/processed"
EXCEL = ROOT / "data/excel"


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


def build_feature_dataset(
    mode: str,
    annotations: pd.DataFrame,
    excluded: set[str],
    processed_dir: Path,
    output: Path,
    preprocessing: str = "bandpass",
) -> tuple[list[dict], dict]:
    recordings = sorted(set(annotations["recording"]) - excluded)
    recording_files = [processed_dir / f"{recording}.parquet" for recording in recordings]
    missing = [path for path in recording_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing converted files: {missing}")

    temporary = output.with_name(output.stem + ".tmp.parquet")
    writer = None
    report_rows = []
    try:
        for path in recording_files:
            recording = path.stem
            frame = pd.read_parquet(path, engine="pyarrow")
            required = ["Time", *CHANNELS, "Class"]
            if frame.columns.tolist() != required:
                raise ValueError(f"{recording}: unexpected columns {frame.columns.tolist()}")
            signal = preprocess(frame.loc[:, CHANNELS].to_numpy(dtype=np.float32, copy=False), preprocessing)
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
            result = pd.concat([metadata, features], axis=1)
            result["class"] = segmentation.classes
            table = pa.Table.from_pandas(result, preserve_index=False)
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
                f"{mode}/{preprocessing}/{recording}: windows={total:,}, class1={class_1:,}, "
                f"represented seizures={represented}/{seizure_count}",
                flush=True,
            )
            del frame, signal, labels, windows, features, result, table
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("No recordings were written")
    temporary.replace(output)

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
    return report_rows, overall


def provisional_reference() -> tuple[list[dict], dict] | None:
    report_path = ROOT / "outputs/window_counts.csv"
    if report_path.exists():
        frame = pd.read_csv(report_path)
        rows = frame.loc[frame["recording"] != "OVERALL"].to_dict("records")
        overall_rows = frame.loc[frame["recording"] == "OVERALL"]
        if not overall_rows.empty:
            return rows, overall_rows.iloc[0].to_dict()
    dataset = PROCESSED / "windows_features.parquet"
    if not dataset.exists():
        return None
    frame = pd.read_parquet(dataset, columns=["recording", "class"])
    rows = []
    for recording, group in frame.groupby("recording"):
        total = len(group)
        class_1 = int(group["class"].sum())
        rows.append({"recording": recording, "total_windows": total, "class_0": total - class_1, "class_1": class_1})
    total = len(frame)
    class_1 = int(frame["class"].sum())
    return rows, {"recording": "OVERALL", "total_windows": total, "class_0": total - class_1, "class_1": class_1}


def compare_reports(official_rows: list[dict], official_overall: dict) -> dict | None:
    reference = provisional_reference()
    if reference is None:
        return None
    provisional_rows, provisional_overall = reference
    official = {row["recording"]: row for row in official_rows}
    provisional = {row["recording"]: row for row in provisional_rows}
    rows = []
    for recording in sorted(set(official) | set(provisional)):
        new = official.get(recording)
        old = provisional.get(recording)
        new_windows = 0 if new is None else int(new["total_windows"])
        old_windows = 0 if old is None else int(old["total_windows"])
        new_class_1 = 0 if new is None else int(new["class_1"])
        old_class_1 = 0 if old is None else int(old["class_1"])
        rows.append(
            {
                "recording": recording,
                "provisional_windows": old_windows,
                "official_windows": new_windows,
                "delta_windows": new_windows - old_windows,
                "provisional_class_1": old_class_1,
                "official_class_1": new_class_1,
                "delta_class_1": new_class_1 - old_class_1,
            }
        )
    return {
        "per_recording": rows,
        "overall": {
            "provisional_windows": int(provisional_overall["total_windows"]),
            "official_windows": int(official_overall["total_windows"]),
            "delta_windows": int(official_overall["total_windows"] - provisional_overall["total_windows"]),
            "provisional_class_1": int(provisional_overall["class_1"]),
            "official_class_1": int(official_overall["class_1"]),
            "delta_class_1": int(official_overall["class_1"] - provisional_overall["class_1"]),
        },
    }


def write_csv(path: Path, rows: list[dict], overall: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(overall))
        writer.writeheader()
        writer.writerows(rows + [overall])


def write_decisions(mode: str, resolution: dict) -> None:
    if mode != "official":
        return
    lines = [
        "# Annotation decisions",
        "",
        "Generated from `config/annotation_decisions.yaml`. Edit that file after the professor replies, then rerun `python scripts/build_dataset.py --annotations official`.",
        "",
        "The official rule is `elapsed = clock annotation − DebutTrace`, with midnight rollover when needed. Raw workbooks are never modified.",
        "",
        "## Applied decisions",
        "",
    ]
    lines.extend(f"- {decision}" for decision in resolution["decisions"])
    lines += ["", "## Signal scores used by configurable decisions", ""]
    for key, values in resolution["scores"].items():
        lines += [f"### {key}", "", "| Source row | Score | Inside/outside ratio | Inside windows |", "|---:|---:|---:|---:|"]
        for value in values:
            row = value.get("source_row", "—")
            lines.append(f"| {row} | {value['score']:.6f} | {value['inside_outside_ratio']:.3f} | {value['inside_windows']} |")
        lines.append("")
    lines += ["## Exclusions", ""]
    lines.extend(f"- `{recording}`: {reason}" for recording, reason in resolution["excluded"].items())
    lines += [
        "",
        "Records marked `fallback` in `data/interim/official/offsets.csv` use provisional intervals because they are absent from the updated workbook. All other included rows use method `official`.",
        "",
    ]
    (ROOT / "docs/ANNOTATION_DECISIONS.md").write_text("\n".join(lines))


def write_report(mode: str, summary: dict, suffix: str) -> None:
    rows = summary["per_recording"]
    overall = summary["overall"]
    preprocessing = summary["parameters"]["preprocessing"]
    if preprocessing["method"] == "bandpass":
        low, high = preprocessing["band_hz"]
        filter_text = (
            f"Before segmentation, each channel of the full recording is band-pass filtered {low:g}–{high:g} Hz "
            f"(Butterworth order {preprocessing['order']}, zero-phase `sosfiltfilt`). This removes the per-recording DC offset; "
            "`mean` features are therefore close to 0 and uninformative but kept as required by the assignment."
        )
    else:
        filter_text = "No filtering is applied: features are computed on the raw signal, including each recording's DC offset."
    lines = [
        f"# Dataset report — {mode.capitalize()} annotations" + (" (unfiltered)" if suffix else ""),
        "",
        "The dataset contains 2 s windows (512 samples at 256 Hz) with a 1 s hop. A window is positive when at least 256 samples are inside a seizure interval. "
        f"Features use the {len(CHANNELS)} EEG channels {', '.join(CHANNELS)}; ECG, EMG and SLI are not used.",
        "",
        filter_text,
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
    comparison = summary.get("comparison_with_provisional")
    if comparison:
        value = comparison["overall"]
        lines += [
            "",
            "## Comparison with the preserved provisional dataset",
            "",
            f"The official build has {value['official_windows']} windows versus {value['provisional_windows']} provisional ({value['delta_windows']:+d}), and {value['official_class_1']} positive windows versus {value['provisional_class_1']} ({value['delta_class_1']:+d}).",
            "",
            "| Recording | Provisional windows | Official windows | Δ windows | Provisional class 1 | Official class 1 | Δ class 1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in comparison["per_recording"]:
            lines.append(
                f"| {row['recording']} | {row['provisional_windows']} | {row['official_windows']} | {row['delta_windows']:+d} | "
                f"{row['provisional_class_1']} | {row['official_class_1']} | {row['delta_class_1']:+d} |"
            )
    lines += [
        "",
        "Features comprise per-channel and channel-average time statistics, absolute Delta/Theta/Alpha/Beta powers, and relative band powers. This report covers dataset construction only; models are documented separately.",
        "",
    ]
    (ROOT / f"docs/DATASET_PARTS_I_VI_{mode.upper()}{suffix.upper()}.md").write_text("\n".join(lines))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--annotations", choices=["official", "provisional"], required=True)
    parser.add_argument("--decisions", help="Optional path to annotation_decisions.yaml")
    parser.add_argument("--preprocess", choices=PREPROCESSING, default="bandpass", help="Signal preprocessing before segmentation")
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Reuse existing labeled Parquet files (labels are checked against the annotations)",
    )
    args = parser.parse_args()
    mode = args.annotations
    # The filtered dataset is the default; the unfiltered variant is kept alongside with a _raw suffix.
    suffix = "" if args.preprocess == "bandpass" else "_raw"
    interim_dir = ROOT / f"data/interim/{mode}"
    excel_dir = EXCEL / mode
    processed_dir = PROCESSED / mode
    output = PROCESSED / f"windows_features_{mode}{suffix}.parquet"
    resolution = resolve_annotations(ROOT, mode, interim_dir, args.decisions)
    annotations = resolution["annotations"]
    excluded = set(resolution["excluded"])
    recordings = sorted(set(annotations["recording"]) - excluded)

    for recording in recordings:
        if args.skip_conversion:
            path = processed_dir / f"{recording}.parquet"
            labels = pd.read_parquet(path, columns=["Class"])["Class"].to_numpy()
            expected = sample_labels(len(labels), annotations_for_recording(annotations, recording))
            if not np.array_equal(labels, expected):
                raise ValueError(f"{path} labels differ from current annotations; rerun without --skip-conversion")
            continue
        convert_recording(recording, force=True, annotations=annotations, excel_dir=excel_dir, processed_dir=processed_dir)

    report_rows, overall = build_feature_dataset(mode, annotations, excluded, processed_dir, output, args.preprocess)
    comparison = compare_reports(report_rows, overall) if mode == "official" else None
    report_csv = ROOT / f"outputs/window_counts_{mode}{suffix}.csv"
    report_json = ROOT / f"outputs/dataset_summary_{mode}{suffix}.json"
    write_csv(report_csv, report_rows, overall)
    summary = {
        "annotation_mode": mode,
        "parameters": {
            "sampling_frequency_hz": 256,
            "window_seconds": 2,
            "window_samples": 512,
            "overlap_percent": 50,
            "hop_samples": 256,
            "positive_rule": "at least 50% (256/512 samples) labeled seizure",
            "channels": list(CHANNELS),
            "preprocessing": describe(args.preprocess),
            "excluded_recordings": sorted(excluded),
        },
        "per_recording": report_rows,
        "overall": overall,
        "comparison_with_provisional": comparison,
        "dataset_columns": pq.read_schema(output).names,
    }
    report_json.write_text(json.dumps(summary, indent=2) + "\n")
    write_decisions(mode, resolution)
    write_report(mode, summary, suffix)
    print(f"Dataset: {output}")
    print(f"Overall: {overall}")


if __name__ == "__main__":
    main()
