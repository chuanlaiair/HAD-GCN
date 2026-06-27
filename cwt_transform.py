"""Morlet CWT and three-channel time-frequency image construction."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import numpy as np
from scipy.ndimage import zoom
from scipy.signal import fftconvolve

LOGGER = logging.getLogger(__name__)


def _morlet_kernel(
    frequency_hz: float,
    sampling_rate: float,
    n_cycles: float,
) -> np.ndarray:
    sigma_t = n_cycles / (2.0 * np.pi * frequency_hz)
    half_duration = max(3.5 * sigma_t, 2.0 / sampling_rate)
    times = np.arange(
        -half_duration,
        half_duration + 1.0 / sampling_rate,
        1.0 / sampling_rate,
        dtype=np.float64,
    )
    wavelet = np.exp(2j * np.pi * frequency_hz * times)
    wavelet *= np.exp(-(times ** 2) / (2.0 * sigma_t ** 2))
    norm = np.sqrt(np.sum(np.abs(wavelet) ** 2))
    return wavelet / max(norm, np.finfo(np.float64).eps)


def morlet_scalogram(
    signal: np.ndarray,
    sampling_rate: float,
    frequencies_hz: np.ndarray,
    n_cycles: float,
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - signal.mean()
    standard_deviation = signal.std()
    if standard_deviation > 0:
        signal = signal / standard_deviation

    rows = []
    for frequency in frequencies_hz:
        kernel = _morlet_kernel(
            float(frequency),
            sampling_rate,
            n_cycles,
        )
        coefficients = fftconvolve(
            signal,
            kernel.conj()[::-1],
            mode="same",
        )
        rows.append(np.abs(coefficients) ** 2)

    return np.log1p(np.asarray(rows, dtype=np.float64))[::-1]


def _robust_normalize(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, [1.0, 99.0])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    image = np.clip(image, low, high)
    return ((image - low) / (high - low)).astype(np.float32)


def _resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    resized = zoom(
        image,
        (height / image.shape[0], width / image.shape[1]),
        order=1,
        mode="nearest",
        prefilter=False,
    )
    output = np.zeros((height, width), dtype=np.float32)
    copy_height = min(height, resized.shape[0])
    copy_width = min(width, resized.shape[1])
    output[:copy_height, :copy_width] = resized[:copy_height, :copy_width]
    return output


def trial_to_tfi(
    trial: np.ndarray,
    sampling_rate: float,
    frequency_range_hz: Tuple[float, float],
    frequency_bins: int,
    n_cycles: float,
    image_size: int,
) -> np.ndarray:
    """Convert C3/Cz/C4 signals to one vertically merged image."""
    if trial.shape[0] != 3:
        raise ValueError("CWT branch requires exactly C3, Cz, and C4.")

    frequencies = np.linspace(
        frequency_range_hz[0],
        frequency_range_hz[1],
        frequency_bins,
        dtype=np.float64,
    )
    heights = [image_size // 3] * 3
    for index in range(image_size - sum(heights)):
        heights[index] += 1

    channel_images = []
    for channel_index, target_height in enumerate(heights):
        scalogram = morlet_scalogram(
            trial[channel_index],
            sampling_rate,
            frequencies,
            n_cycles,
        )
        channel_images.append(
            _resize(
                _robust_normalize(scalogram),
                target_height,
                image_size,
            )
        )

    merged = np.concatenate(channel_images, axis=0)
    return merged[None].astype(np.float32)


def build_tfi_dataset(
    trials: np.ndarray,
    sampling_rate: float,
    frequency_range_hz: Tuple[float, float],
    frequency_bins: int,
    n_cycles: float,
    image_size: int,
    workers: int = 0,
) -> np.ndarray:
    kwargs = {
        "sampling_rate": sampling_rate,
        "frequency_range_hz": frequency_range_hz,
        "frequency_bins": frequency_bins,
        "n_cycles": n_cycles,
        "image_size": image_size,
    }

    def convert(trial: np.ndarray) -> np.ndarray:
        return trial_to_tfi(trial, **kwargs)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            images = list(executor.map(convert, trials))
    else:
        images = []
        for index, trial in enumerate(trials, start=1):
            images.append(convert(trial))
            if index == 1 or index % 25 == 0 or index == len(trials):
                LOGGER.info(
                    "CWT conversion: %d/%d",
                    index,
                    len(trials),
                )

    return np.stack(images, axis=0).astype(np.float32)
