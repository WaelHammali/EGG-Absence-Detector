"""Signal preprocessing applied to full recordings before segmentation."""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from src.io import FS


BANDPASS_HZ = (0.5, 40.0)
BANDPASS_ORDER = 4
PREPROCESSING = ("bandpass", "none")


def bandpass_sos(fs: float = FS, band: tuple[float, float] = BANDPASS_HZ, order: int = BANDPASS_ORDER) -> np.ndarray:
    return butter(order, band, btype="bandpass", fs=fs, output="sos")


def bandpass(signal: np.ndarray, fs: float = FS, band: tuple[float, float] = BANDPASS_HZ, order: int = BANDPASS_ORDER) -> np.ndarray:
    """Zero-phase Butterworth band-pass of a (samples, channels) array, channel by channel.

    Forward-backward filtering doubles the effective order and removes phase lag, so
    seizure boundaries are not shifted. Computed in float64, returned as float32.
    """
    sos = bandpass_sos(fs, band, order)
    return sosfiltfilt(sos, np.asarray(signal, dtype=np.float64), axis=0).astype(np.float32)


def preprocess(signal: np.ndarray, method: str, fs: float = FS) -> np.ndarray:
    if method == "bandpass":
        return bandpass(signal, fs)
    if method == "none":
        return signal
    raise ValueError(f"Unknown preprocessing {method!r}; expected one of {PREPROCESSING}")


def describe(method: str) -> dict:
    if method == "none":
        return {"method": "none"}
    return {
        "method": "bandpass",
        "filter": "Butterworth, zero-phase (scipy.signal.sosfiltfilt)",
        "band_hz": list(BANDPASS_HZ),
        "order": BANDPASS_ORDER,
        "applied_to": "each channel of the full recording, before segmentation",
    }
