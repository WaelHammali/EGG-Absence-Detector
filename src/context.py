"""Recording context taken from the technician notes and the reference channels.

Used only to interpret detections (error analysis). None of this is a model feature.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.io import FS


HPN_GAP_S = 60.0  # notes more than this apart belong to different hyperventilation periods
HPN_TAIL_S = 60.0  # the effect of hyperventilation outlasts the last note
SLI_STEP_S = 20.0  # longest time one flash frequency is assumed to last


def hpn_periods(events: pd.DataFrame) -> list[tuple[float, float]]:
    """Hyperventilation periods of one recording: runs of HPN notes, plus a tail."""
    times = np.sort(events.loc[events["category"] == "HPN", "time_s"].to_numpy(dtype=float))
    periods: list[list[float]] = []
    for time in times:
        if periods and time - periods[-1][1] <= HPN_GAP_S:
            periods[-1][1] = time
        else:
            periods.append([time, time])
    return [(start, end + HPN_TAIL_S) for start, end in periods]


def sli_periods(events: pd.DataFrame) -> list[tuple[float, float, float]]:
    """Photic stimulation steps of one recording as (start, end, flash frequency in Hz).

    Each "SLI=xHz" note starts a step that lasts until the next SLI note, at most SLI_STEP_S.
    """
    notes = events.loc[events["category"] == "SLI"].sort_values("time_s")
    times = notes["time_s"].to_numpy(dtype=float)
    steps = []
    for index, (time, label) in enumerate(zip(times, notes["label"])):
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*hz", label.lower())
        if not match:
            continue
        following = times[index + 1] if index + 1 < len(times) else np.inf
        steps.append((time, min(following, time + SLI_STEP_S), float(match.group(1).replace(",", "."))))
    return steps


def overlapping(start: float, end: float, periods) -> list:
    return [period for period in periods if period[0] < end and period[1] > start]


def sli_channel_active(sli: np.ndarray, start: float, end: float, margin: float = 1.0) -> bool:
    """True when the SLI marker channel pulses between start and end (± margin)."""
    first, last = max(0, int((start - margin) * FS)), int((end + margin) * FS)
    level = np.abs(sli).max()
    return bool(level > 0 and (np.abs(sli[first:last]) > 0.5 * level).any())


def emg_ratio(emg: np.ndarray, start: float, end: float) -> float:
    """RMS of the EMG inside [start, end] relative to its typical 2 s RMS over the recording."""
    block = int(2 * FS)
    usable = emg[: len(emg) // block * block].reshape(-1, block)
    typical = np.median(np.sqrt(np.mean((usable - usable.mean(axis=1, keepdims=True)) ** 2, axis=1)))
    segment = emg[int(start * FS) : int(end * FS)]
    if not len(segment) or typical <= 0:
        return float("nan")
    return float(np.sqrt(np.mean((segment - segment.mean()) ** 2)) / typical)
