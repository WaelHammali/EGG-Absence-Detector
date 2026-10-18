"""Raw-signal augmentation transforms for the classical training pipeline.

Each function takes windows shaped (n, samples, channels) -- the same raw-signal shape
``src.features.extract_features`` consumes -- and a ``numpy.random.Generator``, and returns a
new array of the same shape. Each one is applied once per real seizure window by
scripts/build_augmented.py, the same structural role its ``mirror_order()`` channel swap
already plays for the shift+mirror method.
"""

import numpy as np


def scale_amplitude(windows: np.ndarray, rng: np.random.Generator, low: float = 0.6, high: float = 1.6) -> np.ndarray:
    """Multiply every window by its own random gain factor."""
    factors = rng.uniform(low, high, size=(windows.shape[0], 1, 1)).astype(windows.dtype)
    return windows * factors


def add_noise(windows: np.ndarray, rng: np.random.Generator, relative_sigma: float = 0.05) -> np.ndarray:
    """Add Gaussian noise scaled to each window's own per-channel standard deviation."""
    sigma = windows.std(axis=1, keepdims=True) * relative_sigma
    noise = rng.standard_normal(windows.shape).astype(windows.dtype) * sigma
    return windows + noise


def mask_time(windows: np.ndarray, rng: np.random.Generator, max_fraction: float = 0.25) -> np.ndarray:
    """Zero a random contiguous time span, up to ``max_fraction`` of the window, per window."""
    result = windows.copy()
    samples = windows.shape[1]
    span = max(1, int(round(samples * max_fraction)))
    for index in range(windows.shape[0]):
        length = int(rng.integers(1, span + 1))
        start = int(rng.integers(0, samples - length + 1))
        result[index, start:start + length, :] = 0.0
    return result


def dropout_channels(windows: np.ndarray, rng: np.random.Generator, min_channels: int = 1, max_channels: int = 3) -> np.ndarray:
    """Zero 1 to 3 random channels per window."""
    result = windows.copy()
    channels = windows.shape[2]
    for index in range(windows.shape[0]):
        count = int(rng.integers(min_channels, max_channels + 1))
        for channel in rng.choice(channels, size=count, replace=False):
            result[index, :, channel] = 0.0
    return result
