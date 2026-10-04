"""Infer clock-to-elapsed offsets without modifying raw EEG files.

The search uses 2 s Hann windows every 0.25 s.  Its per-window seizure score is
the mean, over the eight common EEG channels, of 2.5--4 Hz power divided by
0.5--30 Hz power.  Offset candidates are constrained so every supplied
interval fits in the recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matio import load_from_mat
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
DOCS = ROOT / "docs"
INTERIM = ROOT / "data/interim"
FIGURES = ROOT / "outputs/figures/offset_alignment"

CHANNELS = ["EEGFp1", "EEGFp2", "EEGC3", "EEGC4", "EEGT3", "EEGT4", "EEGO1", "EEGO2"]
PLOT_CHANNELS = ["EEGFp1", "EEGC3", "EEGO1"]
FS = 256.0
WINDOW_SECONDS = 2.0
HOP_SECONDS = 0.25
RANDOM_STATE = 42  # Kept explicit for the project's reproducibility convention.
LOW_CONFIDENCE_RATIO = 1.20
PEAK_SEPARATION_SECONDS = 5.0


@dataclass
class Interval:
    start_clock: float
    end_clock: float
    source_row: int
    source_number: int
    note: str = ""


@dataclass
class SearchResult:
    offset: float
    best_score: float
    second_offset: float | None
    second_score: float | None
    peak_ratio: float
    offsets: np.ndarray
    scores: np.ndarray
    window_times: np.ndarray
    seizure_score: np.ndarray


def clock_text(seconds: float) -> str:
    seconds %= 86400
    hours = int(seconds // 3600)
    minutes = int(seconds % 3600 // 60)
    secs = seconds % 60
    if abs(secs - round(secs)) < 1e-7:
        return f"{hours:02d}:{minutes:02d}:{int(round(secs)):02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def raw_intervals(summary: dict) -> dict[str, list[list[Interval]]]:
    """Return every workbook version for each recording."""
    versions: dict[str, list[list[Interval]]] = {}
    rows = summary["annotations"]["Feuil1"]["rows"]
    for row in rows:
        intervals = [
            Interval(
                value["start_day"] * 86400,
                value["end_day"] * 86400,
                row["excel_row"],
                value["number"],
            )
            for value in row["intervals"]
        ]
        versions.setdefault(row["Recording"], []).append(intervals)
    return versions


def audit_d_metadata(recording_ids: list[str]) -> list[dict]:
    rows = []
    for recording in recording_ids:
        obj = load_from_mat(RAW / f"{recording}_0000d.mat", raw_data=True)["output_d"]
        timetable = obj.properties["any"][0, 0]
        fields = list(timetable.dtype.names)
        row_times = timetable["rowTimes"]
        props = row_times.properties
        millis = np.asarray(props["millis"]).reshape(-1)
        rows.append(
            {
                "recording": recording,
                "timetable_class": obj.classname,
                "fields": fields,
                "row_times_class": row_times.classname,
                "row_times_format": np.asarray(props.get("fmt", [])).reshape(-1).tolist(),
                "row_times_count": int(millis.size),
                "row_times_first_ms": float(millis[0]),
                "row_times_last_ms": float(millis[-1]),
                "row_times_step_ms": float(np.median(np.diff(millis))),
                "has_StartTime": "StartTime" in fields,
                "has_SampleRate": "SampleRate" in fields,
                "has_TimeStep": "TimeStep" in fields,
                "array_description": np.asarray(timetable["arrayProps"]["Description"][0, 0]).reshape(-1).tolist(),
                "array_user_data_shape": list(np.asarray(timetable["arrayProps"]["UserData"][0, 0]).shape),
                "custom_properties_empty": int(np.asarray(timetable["CustomProps"]).size) == 1
                and np.asarray(timetable["CustomProps"]).reshape(-1)[0] is None,
            }
        )
    return rows


def load_signal(recording: str) -> np.ndarray:
    table = load_from_mat(RAW / f"{recording}_0000d.mat")["output_d"]
    missing = [channel for channel in CHANNELS if channel not in table]
    if missing:
        raise ValueError(f"{recording}: missing common channels {missing}")
    columns = []
    for channel in CHANNELS:
        columns.append(np.concatenate([np.asarray(block).reshape(-1) for block in table[channel]]))
    return np.column_stack(columns).astype(np.float32, copy=False)


def relative_band_power(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    window_samples = int(FS * WINDOW_SECONDS)
    hop_samples = int(FS * HOP_SECONDS)
    n_windows = 1 + (len(signal) - window_samples) // hop_samples
    hann = np.hanning(window_samples).astype(np.float32)
    freqs = np.fft.rfftfreq(window_samples, d=1 / FS)
    target = (freqs >= 2.5) & (freqs <= 4.0)
    total = (freqs >= 0.5) & (freqs <= 30.0)
    output = np.empty(n_windows, dtype=np.float32)
    sample_axis = np.arange(window_samples)
    for first in range(0, n_windows, 512):
        last = min(first + 512, n_windows)
        starts = np.arange(first, last) * hop_samples
        windows = signal[starts[:, None] + sample_axis[None, :], :]
        windows = windows - windows.mean(axis=1, keepdims=True)
        spectrum = np.fft.rfft(windows * hann[None, :, None], axis=1)
        power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
        ratios = power[:, target, :].sum(axis=1) / np.maximum(power[:, total, :].sum(axis=1), 1e-20)
        output[first:last] = ratios.mean(axis=1)
    centers = WINDOW_SECONDS / 2 + np.arange(n_windows) * HOP_SECONDS
    return centers, output


def interval_score(
    offset: float,
    intervals: list[Interval],
    times: np.ndarray,
    prefix: np.ndarray,
) -> tuple[float, float, int]:
    inside_sum = 0.0
    inside_count = 0
    for interval in intervals:
        start = interval.start_clock - offset
        end = interval.end_clock - offset
        left = int(np.searchsorted(times, start, side="left"))
        right = int(np.searchsorted(times, end, side="right"))
        inside_sum += float(prefix[right] - prefix[left])
        inside_count += right - left
    total_count = len(times)
    if inside_count == 0 or inside_count >= total_count:
        return -np.inf, np.nan, inside_count
    inside_mean = inside_sum / inside_count
    outside_mean = (float(prefix[-1]) - inside_sum) / (total_count - inside_count)
    return inside_mean - outside_mean, inside_mean / max(outside_mean, 1e-20), inside_count


def search_offsets(
    intervals: list[Interval], duration: float, times: np.ndarray, seizure_score: np.ndarray
) -> SearchResult:
    lower = max(value.end_clock for value in intervals) - duration
    upper = min(value.start_clock for value in intervals)
    first = math.ceil(lower / HOP_SECONDS) * HOP_SECONDS
    last = math.floor(upper / HOP_SECONDS) * HOP_SECONDS
    if first > last:
        raise ValueError(f"No feasible offset: [{lower}, {upper}]")
    offsets = np.arange(first, last + HOP_SECONDS / 2, HOP_SECONDS)
    prefix = np.r_[0.0, np.cumsum(seizure_score, dtype=np.float64)]
    scores = np.array([interval_score(value, intervals, times, prefix)[0] for value in offsets])
    best_index = int(np.nanargmax(scores))

    distance = max(1, round(PEAK_SEPARATION_SECONDS / HOP_SECONDS))
    peak_indices, _ = find_peaks(scores, distance=distance)
    peak_indices = np.unique(np.r_[peak_indices, 0, len(scores) - 1, best_index])
    separated = peak_indices[np.abs(peak_indices - best_index) >= distance]
    if separated.size:
        second_index = int(separated[np.argmax(scores[separated])])
        second_score = float(scores[second_index])
        second_offset = float(offsets[second_index])
        best_score = float(scores[best_index])
        if second_score > 0:
            ratio = best_score / second_score
        elif best_score > 0:
            ratio = math.inf
        else:
            ratio = 0.0
    else:
        second_offset = None
        second_score = None
        ratio = math.inf
    return SearchResult(
        offset=float(offsets[best_index]),
        best_score=float(scores[best_index]),
        second_offset=second_offset,
        second_score=second_score,
        peak_ratio=float(ratio),
        offsets=offsets,
        scores=scores,
        window_times=times,
        seizure_score=seizure_score,
    )


def seizure_markers(events: list[dict]) -> list[dict]:
    """Extract explicit seizure markers and collapse exact a/b pairs to centers."""
    generic = []
    a_events = []
    b_events = []
    for event in events:
        text = event["text"].strip()
        folded = text.casefold()
        if folded == "a":
            a_events.append(event["onset_s"])
        elif folded == "b":
            b_events.append(event["onset_s"])
        elif re.search(r"(?i)(abs\w*nce|crise|pointe[ -]?onde|(^|\W)po($|\W))", text):
            generic.append({"elapsed": event["onset_s"], "label": text})
    used_b: set[int] = set()
    for start in a_events:
        candidates = [(i, end) for i, end in enumerate(b_events) if i not in used_b and end > start and end - start <= 60]
        if candidates:
            index, end = min(candidates, key=lambda pair: pair[1] - start)
            used_b.add(index)
            generic.append({"elapsed": (start + end) / 2, "label": f"a/b center ({start:g}–{end:g} s)"})
    return sorted(generic, key=lambda value: value["elapsed"])


def header_offset(markers: list[dict], intervals: list[Interval], duration: float) -> dict | None:
    if not markers:
        return None
    midpoints = np.array([(value.start_clock + value.end_clock) / 2 for value in intervals])
    lower = max(value.end_clock for value in intervals) - duration
    upper = min(value.start_clock for value in intervals)
    anchors = []
    for marker in markers:
        for midpoint in midpoints:
            candidate = midpoint - marker["elapsed"]
            if lower <= candidate <= upper:
                anchors.append(candidate)
    if not anchors:
        return None
    candidates = np.unique(np.round(np.asarray(anchors) / HOP_SECONDS) * HOP_SECONDS)
    evaluations = []
    for candidate in candidates:
        residuals = np.array(
            [np.min(np.abs(midpoints - (marker["elapsed"] + candidate))) for marker in markers]
        )
        support = int((residuals <= 5.0).sum())
        supported_error = float(residuals[residuals <= 5.0].mean()) if support else math.inf
        evaluations.append((support, supported_error, float(np.median(residuals)), float(candidate), residuals))
    support, mean_error, median_error, candidate, residuals = min(
        evaluations, key=lambda value: (-value[0], value[1], value[2])
    )
    return {
        "offset": candidate,
        "feasible_offset_lower": lower,
        "feasible_offset_upper": upper,
        "all_excel_intervals_in_recording": True,
        "support": support,
        "marker_count": len(markers),
        "mean_supported_error": mean_error,
        "median_error": median_error,
        "residuals": residuals.tolist(),
        "markers": markers,
    }


def markers_compatible_with_offset(
    offset: float, markers: list[dict], intervals: list[Interval], tolerance: float = 2.0
) -> bool | None:
    """Check whether every header marker lands in some seizure, allowing tolerance."""
    if not markers:
        return None
    for marker in markers:
        clock_at_marker = marker["elapsed"] + offset
        distances = [
            0.0
            if interval.start_clock <= clock_at_marker <= interval.end_clock
            else min(abs(clock_at_marker - interval.start_clock), abs(clock_at_marker - interval.end_clock))
            for interval in intervals
        ]
        if min(distances) > tolerance:
            return False
    return True


def plot_alignment(
    recording: str,
    signal: np.ndarray,
    intervals: list[Interval],
    result: SearchResult,
    header: dict | None,
) -> None:
    rows = len(intervals) + 1
    figure, axes = plt.subplots(rows, 1, figsize=(15, 3.0 * rows), constrained_layout=True)
    if rows == 1:
        axes = [axes]
    axis = axes[0]
    axis.plot(result.offsets, result.scores, color="navy", linewidth=0.9)
    axis.axvline(result.offset, color="crimson", linewidth=1.5, label=f"best data offset {result.offset:.2f} s")
    if result.second_offset is not None:
        axis.axvline(result.second_offset, color="darkorange", linestyle="--", label=f"second peak {result.second_offset:.2f} s")
    if header is not None:
        axis.axvline(header["offset"], color="forestgreen", linestyle=":", linewidth=1.8, label=f"header offset {header['offset']:.2f} s")
    axis.set(
        title=f"{recording}: data-driven clock offset search",
        xlabel="Offset = clock seconds − elapsed seconds (s)",
        ylabel="Inside − outside relative 2.5–4 Hz power",
    )
    axis.legend(loc="best")

    elapsed = np.arange(len(signal)) / FS
    channel_indices = [CHANNELS.index(channel) for channel in PLOT_CHANNELS]
    for seizure_number, (axis, interval) in enumerate(zip(axes[1:], intervals), 1):
        start = interval.start_clock - result.offset
        end = interval.end_clock - result.offset
        view_start = max(0.0, start - 10)
        view_end = min(len(signal) / FS, end + 10)
        left = max(0, int(math.floor(view_start * FS)))
        right = min(len(signal), int(math.ceil(view_end * FS)))
        segment = signal[left:right, channel_indices].astype(np.float64)
        scale = np.nanmedian(np.std(segment, axis=0))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        offsets = np.array([6.0, 3.0, 0.0])
        for column, (channel, vertical) in enumerate(zip(PLOT_CHANNELS, offsets)):
            axis.plot(elapsed[left:right], segment[:, column] / scale + vertical, linewidth=0.55, label=channel)
        axis.axvspan(start, end, color="orange", alpha=0.28)
        axis.set_xlim(view_start, view_end)
        axis.set_yticks(offsets, [value.removeprefix("EEG") for value in PLOT_CHANNELS])
        axis.set(
            title=f"Seizure {seizure_number}: {start:.2f}–{end:.2f} s elapsed ({interval.end_clock - interval.start_clock:.1f} s)",
            xlabel="Elapsed time (s)",
            ylabel="Channels (scaled)",
        )
    figure.savefig(FIGURES / f"{recording}_offset_alignment.png", dpi=140)
    plt.close(figure)


def main() -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = json.loads((DOCS / "inspection_summary.json").read_text())
    recordings = {row["Recording"]: row for row in summary["recordings"]}
    versions = raw_intervals(summary)
    metadata = audit_d_metadata(list(recordings))
    (DOCS / "dfile_time_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")

    # Candidate annotation sets.  The suspected three-hour end typo is evaluated
    # as an explicit candidate, while the raw workbook remains unchanged.
    candidate_sets: dict[str, list[tuple[str, list[Interval]]]] = {}
    for recording, alternatives in versions.items():
        candidate_sets[recording] = [(f"source row {values[0].source_row}" if values else "no intervals", values) for values in alternatives]
    corrected_210406 = [
        Interval(
            versions["210406A_F"][0][0].start_clock,
            12 * 3600 + 15 * 60 + 58,
            versions["210406A_F"][0][0].source_row,
            1,
            "Workbook end 15:15:58 evaluated as suspected hour typo 12:15:58.",
        )
    ]
    candidate_sets["210406A_F"] = [("candidate correction 12:15:58", corrected_210406)]

    result_rows = []
    chosen_intervals: dict[str, list[Interval]] = {}
    chosen_results: dict[str, SearchResult] = {}
    headers: dict[str, dict | None] = {}
    interval_diagnostics: dict[str, list[dict]] = {}
    alternatives_report: dict[str, list[dict]] = {}

    for recording, rec in recordings.items():
        if recording == "200625A_F":
            continue
        signal = load_signal(recording)
        times, power = relative_band_power(signal)
        alternatives = []
        for label, intervals in candidate_sets[recording]:
            if not intervals:
                continue
            result = search_offsets(intervals, rec["duration_s"], times, power)
            alternatives.append((label, intervals, result))
        label, intervals, result = max(alternatives, key=lambda value: value[2].best_score)
        alternatives_report[recording] = [
            {
                "label": value[0],
                "offset": value[2].offset,
                "best_score": value[2].best_score,
                "peak_ratio": value[2].peak_ratio,
            }
            for value in alternatives
        ]
        markers = seizure_markers(rec["header_events"])
        hdr = header_offset(markers, intervals, rec["duration_s"])
        compatible = markers_compatible_with_offset(result.offset, markers, intervals)
        agrees = "NA" if compatible is None else ("yes" if compatible else "no")

        prefix = np.r_[0.0, np.cumsum(power, dtype=np.float64)]
        diagnostics = []
        for interval in intervals:
            _, ratio, count = interval_score(result.offset, [interval], times, prefix)
            diagnostics.append(
                {
                    "source_number": interval.source_number,
                    "elapsed_start": interval.start_clock - result.offset,
                    "elapsed_end": interval.end_clock - result.offset,
                    "inside_outside_ratio": ratio,
                    "window_count": count,
                }
            )
        low = result.peak_ratio < LOW_CONFIDENCE_RATIO or result.best_score <= 0
        result_rows.append(
            {
                "recording": recording,
                "offset_clock": result.offset,
                "start_clock": clock_text(result.offset),
                "method": "data_driven",
                "confidence": result.peak_ratio,
                "agrees_with_header": agrees,
                "low_confidence": "yes" if low else "no",
                "best_score": result.best_score,
                "second_best_offset": result.second_offset,
                "second_best_score": result.second_score,
                "annotation_version": label,
                "header_offset": None if hdr is None else hdr["offset"],
            }
        )
        chosen_intervals[recording] = intervals
        chosen_results[recording] = result
        headers[recording] = hdr
        interval_diagnostics[recording] = diagnostics
        plot_alignment(recording, signal, intervals, result, hdr)
        print(
            f"{recording}: offset={result.offset:.2f}, score={result.best_score:.6f}, "
            f"peak ratio={result.peak_ratio:.3f}, header={None if hdr is None else hdr['offset']}, "
            f"agreement={agrees}, version={label}",
            flush=True,
        )
        del signal, power

    result_rows.sort(key=lambda value: value["recording"])
    with (INTERIM / "offsets.csv").open("w", newline="") as handle:
        fields = ["recording", "offset_clock", "start_clock", "method", "confidence", "agrees_with_header"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in result_rows:
            writer.writerow(row)

    # Apply annotation decisions only in the derived file.
    clean_rows = []
    decision_notes = []
    for row in result_rows:
        recording = row["recording"]
        result = chosen_results[recording]
        intervals = chosen_intervals[recording]
        diagnostics = interval_diagnostics[recording]
        if recording == "210914B_A":
            keep = [value["inside_outside_ratio"] > 1 for value in diagnostics]
            decision_notes.append(
                f"210914B_A: {sum(keep)}/11 intervals have above-background 2.5–4 Hz power at the shared alignment."
            )
        elif recording == "210406A_F":
            confirmed = diagnostics[0]["inside_outside_ratio"] > 1 and row["best_score"] > 0
            if not confirmed:
                intervals = []
            decision_notes.append(
                "210406A_F: suspected 12:15:58 correction "
                + ("retained after positive signal confirmation." if confirmed else "rejected; interval dropped because signal confirmation failed.")
            )
        elif recording == "230515B_G":
            decision_notes.append(f"230515B_G: selected {row['annotation_version']} by the larger constrained data score.")
        for seizure_id, interval in enumerate(intervals, 1):
            note_parts = []
            if interval.note:
                note_parts.append(interval.note)
            if recording == "210914B_A":
                diag = diagnostics[seizure_id - 1]
                note_parts.append(
                    f"All 11 source pairs retained; 2.5–4 Hz inside/outside ratio={diag['inside_outside_ratio']:.3f}."
                )
            if recording == "230515B_G":
                note_parts.append(f"Selected {row['annotation_version']} over the conflicting workbook row.")
            note_parts.append("Clock interval shifted by inferred recording offset.")
            clean_rows.append(
                {
                    "recording": recording,
                    "seizure_id": seizure_id,
                    "start_s": round(interval.start_clock - result.offset, 6),
                    "end_s": round(interval.end_clock - result.offset, 6),
                    "duration_s": round(interval.end_clock - interval.start_clock, 6),
                    "source_row": interval.source_row,
                    "note": " ".join(note_parts),
                }
            )
    with (INTERIM / "annotations_clean.csv").open("w", newline="") as handle:
        fields = ["recording", "seizure_id", "start_s", "end_s", "duration_s", "source_row", "note"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(clean_rows)

    # Intercrit* entries are clock-time point events, explicitly stored as class 0
    # for later analysis.  Use only the workbook row selected for each recording.
    source_rows = {
        recording: intervals[0].source_row
        for recording, intervals in chosen_intervals.items()
        if intervals
    }
    result_by_recording = {row["recording"]: row for row in result_rows}
    intercritical_rows = []
    for annotation_row in summary["annotations"]["Feuil1"]["rows"]:
        recording = annotation_row["Recording"]
        if source_rows.get(recording) != annotation_row["excel_row"]:
            continue
        offset = result_by_recording[recording]["offset_clock"]
        duration = recordings[recording]["duration_s"]
        event_number = 0
        for source_column, clock_days in enumerate(annotation_row["intercritical"], 1):
            if clock_days is None:
                continue
            event_number += 1
            clock_seconds = clock_days * 86400
            elapsed_seconds = clock_seconds - offset
            intercritical_rows.append(
                {
                    "recording": recording,
                    "event_id": event_number,
                    "clock_s": round(clock_seconds, 6),
                    "event_s": round(elapsed_seconds, 6),
                    "Class": 0,
                    "source_row": annotation_row["excel_row"],
                    "source_column": f"Intercrit{source_column}",
                    "inside_recording": "yes" if 0 <= elapsed_seconds <= duration else "no",
                    "note": "Interictal point event; not a seizure interval.",
                }
            )
    with (INTERIM / "intercritical_events.csv").open("w", newline="") as handle:
        fields = [
            "recording", "event_id", "clock_s", "event_s", "Class", "source_row",
            "source_column", "inside_recording", "note",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(intercritical_rows)

    details = {
        "parameters": {
            "channels": CHANNELS,
            "fs_hz": FS,
            "window_seconds": WINDOW_SECONDS,
            "hop_seconds": HOP_SECONDS,
            "target_band_hz": [2.5, 4.0],
            "denominator_band_hz": [0.5, 30.0],
            "second_peak_minimum_separation_seconds": PEAK_SEPARATION_SECONDS,
            "low_confidence_peak_ratio_threshold": LOW_CONFIDENCE_RATIO,
        },
        "results": result_rows,
        "header_anchors": headers,
        "interval_diagnostics": interval_diagnostics,
        "annotation_alternatives": alternatives_report,
        "decision_notes": decision_notes,
    }
    (DOCS / "offset_alignment_details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n")
    write_report(metadata, result_rows, details, clean_rows, intercritical_rows)


def write_report(
    metadata: list[dict], result_rows: list[dict], details: dict,
    clean_rows: list[dict], intercritical_rows: list[dict]
) -> None:
    lines = [
        "# Clock offset alignment",
        "",
        "No raw file was modified and no feature dataset was built. `offset_clock` means Excel clock seconds minus elapsed signal seconds; it is therefore the inferred clock time at elapsed 0 s.",
        "",
        "## D-file timetable metadata",
        "",
        "All 22 d-files have the same relevant structure. The raw MCOS timetable fields are `CustomProps`, `VariableCustomProps`, `versionSavedFrom`, `minCompatibleVersion`, `incompatibilityMsg`, `arrayProps`, `data`, `numDims`, `useVarNamesOrig`, `useDimNamesOrig`, `dimNames`, `dimNamesOrig`, `varNames`, `varNamesOrig`, `numRows`, `numVars`, `varDescriptions`, `varUnits`, `rowTimes`, and `varContinuity`.",
        "",
        "`rowTimes` is a MATLAB `duration`, with `millis = [0, 1000, 2000, …]` and format `s`. It is not an absolute `datetime`. No d-file has a `StartTime`, `SampleRate`, or `TimeStep` field. `Description`, `UserData`, table custom properties, variable descriptions and variable units are empty. The full per-file audit is in [dfile_time_metadata.json](dfile_time_metadata.json). The 256 Hz rate comes from 256 samples stored in each one-second channel block.",
        "",
        "## Data-driven method and confidence",
        "",
        "For each 2 s Hann window at a 0.25 s hop, the seizure score is mean relative 2.5–4 Hz power across Fp1, Fp2, C3, C4, T3, T4, O1 and O2; the denominator is 0.5–30 Hz power. For every feasible offset at 0.25 s resolution, the objective is mean window score inside all shifted seizure intervals minus mean score outside.",
        "",
        f"The second peak must be at least {PEAK_SEPARATION_SECONDS:g} s from the best peak. Confidence is best score divided by second-best score; ratios below {LOW_CONFIDENCE_RATIO:g}, or a non-positive best score, are flagged low. This ratio measures offset uniqueness, not clinical certainty.",
        "",
        "| Recording | Offset (s) | Start clock | Best score | Second peak: offset / score | Ratio | Header offset | Agreement ≤2 s | Low confidence | Annotation version |",
        "|---|---:|---|---:|---|---:|---:|---|---|---|",
    ]
    for row in result_rows:
        second = "—" if row["second_best_offset"] is None else f"{row['second_best_offset']:.2f} / {row['second_best_score']:.6f}"
        header = "—" if row["header_offset"] is None else f"{row['header_offset']:.2f}"
        ratio = "∞" if math.isinf(row["confidence"]) else f"{row['confidence']:.3f}"
        lines.append(
            f"| {row['recording']} | {row['offset_clock']:.2f} | {row['start_clock']} | {row['best_score']:.6f} | {second} | {ratio} | {header} | {row['agrees_with_header']} | {row['low_confidence']} | {row['annotation_version']} |"
        )
    header_available = sum(row["header_offset"] is not None for row in result_rows)
    header_agrees = sum(row["agrees_with_header"] == "yes" for row in result_rows)
    low = [row["recording"] for row in result_rows if row["low_confidence"] == "yes"]
    lines += [
        "",
        f"Seizure-like header anchors produced candidate offsets for **{header_available}** recordings; in **{header_agrees}**, every extracted marker falls inside a shifted Excel interval or within 2 s of a boundary under the independent data optimum. Header extraction includes `absence`, `crise`, `pointe-onde`/`PO`, and centers of explicit `a`/`b` pairs; it excludes unrelated `HPN fin` and export-end events. The displayed header offset is an independent midpoint-match representative; generic event labels can occur anywhere within a seizure, so compatibility is tested against interval ranges rather than requiring that representative point to equal the data optimum. Every header candidate is constrained to the feasible offset interval, which places every Excel seizure inside the recording. Full marker lists, feasible ranges and residuals are in [offset_alignment_details.json](offset_alignment_details.json).",
        "",
        "Low-confidence recordings under the stated ratio rule: " + (", ".join(f"`{value}`" for value in low) if low else "none") + ".",
        "",
        "## Annotation decisions",
        "",
    ]
    lines.extend(f"- {note}" for note in details["decision_notes"])
    lines += [
        "- 200625A_F is absent from both derived CSVs because it is unknown/unannotated, not a confirmed class-0 recording.",
        f"- `{len(intercritical_rows)}` populated `Intercrit*` point events are stored as class 0 in [intercritical_events.csv](../data/interim/intercritical_events.csv). They were not converted to seizure intervals.",
        "- ECG, SLI and EMG are excluded. The fixed EEG order is Fp1, Fp2, C3, C4, T3, T4, O1, O2.",
        "",
        f"The cleaned derived file contains **{len(clean_rows)} seizure intervals**. It is [annotations_clean.csv](../data/interim/annotations_clean.csv); inferred starts are in [offsets.csv](../data/interim/offsets.csv).",
        "",
        "## Plots",
        "",
        "There is one figure per annotated recording in [outputs/figures/offset_alignment](../outputs/figures/offset_alignment). The first panel shows the full objective curve and its two reported peaks. Every following panel shows Fp1, C3 and O1 for one aligned interval with ±10 s context; channels are independently displayed on a shared robust scale and the proposed seizure interval is shaded.",
        "",
        "These are alignment diagnostics. A high 3 Hz score supports an offset, but visual morphology and ambiguous annotations still require review before dataset construction.",
        "",
    ]
    (DOCS / "OFFSET_ALIGNMENT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
