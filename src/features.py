"""Time-domain and Hann-rFFT band-power features for EEG windows."""

import numpy as np
import pandas as pd

from src.io import CHANNELS, FS


TIME_FEATURES = ("mean", "std", "variance", "min", "max", "amplitude", "rms", "energy")
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}
FREQUENCY_FEATURES = tuple(
    name for band in BANDS for name in (f"{band}_power", f"{band}_relative_power")
)
ALL_FEATURES = TIME_FEATURES + FREQUENCY_FEATURES


def _band_powers(windows: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    count, samples, channels = windows.shape
    hann = np.hanning(samples).astype(np.float32)
    frequencies = np.fft.rfftfreq(samples, d=1 / fs)
    df = frequencies[1] - frequencies[0]
    absolute = {band: np.empty((count, channels), dtype=np.float32) for band in BANDS}
    relative = {band: np.empty((count, channels), dtype=np.float32) for band in BANDS}
    total_mask = (frequencies >= 0.5) & (frequencies <= 30.0)
    normalization = fs * np.sum(hann * hann)
    for first in range(0, count, 512):
        last = min(first + 512, count)
        block = np.asarray(windows[first:last], dtype=np.float32)
        centered = block - block.mean(axis=1, keepdims=True)
        spectrum = np.fft.rfft(centered * hann[None, :, None], axis=1)
        psd = (spectrum.real * spectrum.real + spectrum.imag * spectrum.imag) / normalization
        if psd.shape[1] > 2:
            psd[:, 1:-1, :] *= 2.0
        total = np.maximum(psd[:, total_mask, :].sum(axis=1) * df, 1e-20)
        for band, (low, high) in BANDS.items():
            mask = (frequencies >= low) & (frequencies < high)
            if band == "beta":
                mask = (frequencies >= low) & (frequencies <= high)
            values = psd[:, mask, :].sum(axis=1) * df
            absolute[band][first:last] = values.astype(np.float32)
            relative[band][first:last] = (values / total).astype(np.float32)
    output = {}
    for band in BANDS:
        output[f"{band}_power"] = absolute[band]
        output[f"{band}_relative_power"] = relative[band]
    return output


def extract_features(windows: np.ndarray, fs: float = FS) -> pd.DataFrame:
    """Calculate per-channel features and the mean of each feature across channels."""
    values: dict[str, np.ndarray] = {
        "mean": windows.mean(axis=1, dtype=np.float64).astype(np.float32),
        "std": windows.std(axis=1, dtype=np.float64).astype(np.float32),
        "variance": windows.var(axis=1, dtype=np.float64).astype(np.float32),
        "min": windows.min(axis=1),
        "max": windows.max(axis=1),
        "amplitude": np.ptp(windows, axis=1),
        "rms": np.sqrt(np.mean(np.square(windows, dtype=np.float64), axis=1)).astype(np.float32),
        "energy": np.sum(np.square(windows, dtype=np.float64), axis=1).astype(np.float32),
    }
    values.update(_band_powers(windows, fs))
    columns: dict[str, np.ndarray] = {}
    for feature in ALL_FEATURES:
        matrix = values[feature]
        for index, channel in enumerate(CHANNELS):
            columns[f"{channel}_{feature}"] = matrix[:, index]
        columns[f"avg_{feature}"] = matrix.mean(axis=1, dtype=np.float64).astype(np.float32)
    return pd.DataFrame(columns)

