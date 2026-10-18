"""Turn window probabilities into seizure intervals and compare them with annotations."""

from __future__ import annotations

import numpy as np
import pandas as pd


MIN_GAP_S = 2.0
MIN_DURATION_S = 2.0
INTERVAL_COLUMNS = ["start_s", "end_s", "duration_s", "max_proba", "mean_proba"]


def windows_to_intervals(
    start_s: np.ndarray,
    end_s: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    min_gap_s: float = MIN_GAP_S,
    min_duration_s: float = MIN_DURATION_S,
) -> pd.DataFrame:
    """Detected intervals for one recording.

    A window is positive when ``proba >= threshold``. Runs of consecutive positive windows
    form an interval from the start of the first window to the end of the last one; intervals
    separated by less than ``min_gap_s`` are merged, then events shorter than
    ``min_duration_s`` are dropped.
    """
    start_s = np.asarray(start_s, dtype=float)
    end_s = np.asarray(end_s, dtype=float)
    proba = np.asarray(proba, dtype=float)
    order = np.argsort(start_s, kind="stable")
    start_s, end_s, proba = start_s[order], end_s[order], proba[order]
    positive = np.flatnonzero(proba >= threshold)
    if not len(positive):
        return pd.DataFrame(columns=INTERVAL_COLUMNS)

    # Each run is [first window index, last window index].
    breaks = np.flatnonzero(np.diff(positive) > 1)
    runs = [[int(first), int(last)] for first, last in zip(positive[np.r_[0, breaks + 1]], positive[np.r_[breaks, len(positive) - 1]])]
    merged = [runs[0]]
    for first, last in runs[1:]:
        if start_s[first] - end_s[merged[-1][1]] < min_gap_s:
            merged[-1][1] = last
        else:
            merged.append([first, last])
    rows = []
    for first, last in merged:
        duration = end_s[last] - start_s[first]
        if duration < min_duration_s:
            continue
        inside = proba[first : last + 1]
        rows.append(
            {
                "start_s": start_s[first],
                "end_s": end_s[last],
                "duration_s": duration,
                "max_proba": float(inside.max()),
                "mean_proba": float(inside.mean()),
            }
        )
    return pd.DataFrame(rows, columns=INTERVAL_COLUMNS)


def smooth_probabilities(table: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Centred moving median of each recording's probability, ordered by ``start_s``.

    Removes isolated one-window spikes (noise) without retraining the model, at the cost of
    slightly blurring real onsets and offsets. ``window`` must be odd; 1 leaves ``proba``
    unchanged.
    """
    if window <= 1:
        return table
    parts = []
    for _, group in table.groupby("recording", sort=False):
        ordered = group.sort_values("start_s")
        parts.append(ordered.assign(proba=ordered["proba"].rolling(window, center=True, min_periods=1).median()))
    return pd.concat(parts).loc[table.index]


def windows_to_intervals_hysteresis(
    start_s: np.ndarray,
    end_s: np.ndarray,
    proba: np.ndarray,
    high: float,
    low: float,
    min_gap_s: float = MIN_GAP_S,
    min_duration_s: float = MIN_DURATION_S,
) -> pd.DataFrame:
    """Like :func:`windows_to_intervals`, but an event needs ``proba >= high`` to start and only
    ``proba >= low`` to continue (``low <= high``; ``low == high`` is the plain single threshold).
    Lets a confirmed event survive a few weak windows without lowering the bar for starting a
    new one, so isolated noise stays rejected.
    """
    start_s = np.asarray(start_s, dtype=float)
    end_s = np.asarray(end_s, dtype=float)
    proba = np.asarray(proba, dtype=float)
    order = np.argsort(start_s, kind="stable")
    start_s, end_s, proba = start_s[order], end_s[order], proba[order]
    positive = np.zeros(len(proba), dtype=bool)
    active = False
    for index, value in enumerate(proba):
        active = value >= high if not active else value >= low
        positive[index] = active
    # Same run-finding, merging and max/mean-proba bookkeeping as windows_to_intervals, but
    # driven by the hysteresis ``positive`` mask instead of a plain threshold comparison, so
    # max_proba / mean_proba stay the real probabilities, not the binarized signal.
    runs = np.flatnonzero(positive)
    if not len(runs):
        return pd.DataFrame(columns=INTERVAL_COLUMNS)
    breaks = np.flatnonzero(np.diff(runs) > 1)
    segments = [[int(first), int(last)] for first, last in zip(runs[np.r_[0, breaks + 1]], runs[np.r_[breaks, len(runs) - 1]])]
    merged = [segments[0]]
    for first, last in segments[1:]:
        if start_s[first] - end_s[merged[-1][1]] < min_gap_s:
            merged[-1][1] = last
        else:
            merged.append([first, last])
    rows = []
    for first, last in merged:
        duration = end_s[last] - start_s[first]
        if duration < min_duration_s:
            continue
        inside = proba[first : last + 1]
        rows.append({"start_s": start_s[first], "end_s": end_s[last], "duration_s": duration, "max_proba": float(inside.max()), "mean_proba": float(inside.mean())})
    return pd.DataFrame(rows, columns=INTERVAL_COLUMNS)


def detect_all_hysteresis(predictions: pd.DataFrame, annotations: pd.DataFrame, high: float, low: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """:func:`detect_all` with :func:`windows_to_intervals_hysteresis` instead of a single threshold."""
    detected_frames, match_frames = [], []
    for recording, group in predictions.groupby("recording", sort=True):
        detected = windows_to_intervals_hysteresis(group["start_s"], group["end_s"], group["proba"], high, low)
        real = annotations.loc[annotations["recording"] == recording, ["start_s", "end_s"]]
        matches = match_events(detected, real)
        detected.insert(0, "recording", recording)
        matches.insert(0, "recording", recording)
        detected_frames.append(detected)
        match_frames.append(matches)
    return pd.concat(detected_frames, ignore_index=True), pd.concat(match_frames, ignore_index=True)


def match_events(detected: pd.DataFrame, real: pd.DataFrame) -> pd.DataFrame:
    """Event-level comparison for one recording.

    One row per real seizure (status ``TP`` when at least one detection overlaps it, else
    ``FN``) and one row per detection overlapping no real seizure (status ``FP``). Onset and
    offset errors are detected minus real, in seconds, using the union of the detections that
    overlap the seizure: negative onset error = detected early, positive offset error = ends late.
    """
    detected = detected.reset_index(drop=True)
    real = real.sort_values("start_s").reset_index(drop=True)
    used = np.zeros(len(detected), dtype=bool)
    rows = []
    for seizure in real.itertuples(index=False):
        overlap = (
            (detected["start_s"].to_numpy() < seizure.end_s) & (detected["end_s"].to_numpy() > seizure.start_s)
            if len(detected) else np.zeros(0, dtype=bool)
        )
        used |= overlap
        row = {
            "status": "TP" if overlap.any() else "FN",
            "real_start_s": float(seizure.start_s),
            "real_end_s": float(seizure.end_s),
            "detected_start_s": np.nan,
            "detected_end_s": np.nan,
            "onset_error_s": np.nan,
            "offset_error_s": np.nan,
            "max_proba": np.nan,
        }
        if overlap.any():
            hits = detected.loc[overlap]
            row.update(
                {
                    "detected_start_s": float(hits["start_s"].min()),
                    "detected_end_s": float(hits["end_s"].max()),
                    "onset_error_s": float(hits["start_s"].min() - seizure.start_s),
                    "offset_error_s": float(hits["end_s"].max() - seizure.end_s),
                    "max_proba": float(hits["max_proba"].max()),
                }
            )
        rows.append(row)
    for detection in detected.loc[~used].itertuples(index=False):
        rows.append(
            {
                "status": "FP",
                "real_start_s": np.nan,
                "real_end_s": np.nan,
                "detected_start_s": float(detection.start_s),
                "detected_end_s": float(detection.end_s),
                "onset_error_s": np.nan,
                "offset_error_s": np.nan,
                "max_proba": float(detection.max_proba),
            }
        )
    columns = ["status", "real_start_s", "real_end_s", "detected_start_s", "detected_end_s", "onset_error_s", "offset_error_s", "max_proba"]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    frame["sort_s"] = frame["real_start_s"].fillna(frame["detected_start_s"])
    return frame.sort_values("sort_s").drop(columns="sort_s").reset_index(drop=True)


def event_metrics(matches: pd.DataFrame, detected_count: int | None = None) -> dict:
    """Event-level counts and scores from ``match_events`` rows (one or many recordings).

    Recall counts real seizures found. Precision counts detections that overlap a real
    seizure; when ``detected_count`` is omitted it is approximated by TP + FP.
    """
    tp = int((matches["status"] == "TP").sum())
    fn = int((matches["status"] == "FN").sum())
    fp = int((matches["status"] == "FP").sum())
    true_detections = tp if detected_count is None else detected_count - fp
    precision = true_detections / (true_detections + fp) if true_detections + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    found = matches.loc[matches["status"] == "TP"]
    return {
        "tp": tp, "fn": fn, "fp": fp,
        "precision": precision, "recall": recall, "f1": f1,
        "onset_error_mean_s": float(found["onset_error_s"].mean()) if tp else np.nan,
        "onset_error_abs_median_s": float(found["onset_error_s"].abs().median()) if tp else np.nan,
        "offset_error_mean_s": float(found["offset_error_s"].mean()) if tp else np.nan,
        "offset_error_abs_median_s": float(found["offset_error_s"].abs().median()) if tp else np.nan,
    }


def detect_all(predictions: pd.DataFrame, annotations: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run detection and matching for every recording in ``predictions``.

    Returns (detected intervals, event matches), both with a ``recording`` column.
    """
    detected_frames, match_frames = [], []
    for recording, group in predictions.groupby("recording", sort=True):
        detected = windows_to_intervals(group["start_s"], group["end_s"], group["proba"], threshold)
        real = annotations.loc[annotations["recording"] == recording, ["start_s", "end_s"]]
        matches = match_events(detected, real)
        detected.insert(0, "recording", recording)
        matches.insert(0, "recording", recording)
        detected_frames.append(detected)
        match_frames.append(matches)
    return pd.concat(detected_frames, ignore_index=True), pd.concat(match_frames, ignore_index=True)


def sweep_thresholds(predictions: pd.DataFrame, annotations: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        detected, matches = detect_all(predictions, annotations, float(threshold))
        rows.append({"threshold": float(threshold), "detections": len(detected), **event_metrics(matches, len(detected))})
    return pd.DataFrame(rows)
