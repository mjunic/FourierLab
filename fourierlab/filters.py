"""Frequency-domain filter mask generation."""

from __future__ import annotations

import numpy as np


def distance_grid(shape: tuple[int, int]) -> np.ndarray:
    """Return distance-to-center grid for a 2D image shape."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    return np.sqrt((y - crow) ** 2 + (x - ccol) ** 2)


def ideal_lowpass(shape: tuple[int, int], radius: float) -> np.ndarray:
    return (distance_grid(shape) <= max(radius, 0)).astype(np.float64)


def ideal_highpass(shape: tuple[int, int], radius: float) -> np.ndarray:
    return 1.0 - ideal_lowpass(shape, radius)


def ideal_bandpass(shape: tuple[int, int], r1: float, r2: float) -> np.ndarray:
    lo, hi = sorted((max(r1, 0), max(r2, 0)))
    d = distance_grid(shape)
    return ((d >= lo) & (d <= hi)).astype(np.float64)


def ideal_bandreject(shape: tuple[int, int], r1: float, r2: float) -> np.ndarray:
    return 1.0 - ideal_bandpass(shape, r1, r2)


def gaussian_lowpass(shape: tuple[int, int], sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    d2 = distance_grid(shape) ** 2
    return np.exp(-d2 / (2.0 * sigma * sigma))


def gaussian_highpass(shape: tuple[int, int], sigma: float) -> np.ndarray:
    return 1.0 - gaussian_lowpass(shape, sigma)


def butterworth_lowpass(shape: tuple[int, int], radius: float, order: int) -> np.ndarray:
    radius = max(float(radius), 1e-6)
    order = max(int(order), 1)
    d = distance_grid(shape)
    return 1.0 / (1.0 + (d / radius) ** (2 * order))


def butterworth_highpass(shape: tuple[int, int], radius: float, order: int) -> np.ndarray:
    return 1.0 - butterworth_lowpass(shape, radius, order)


def notch_reject(
    shape: tuple[int, int],
    offset_x: int,
    offset_y: int,
    radius: float,
) -> np.ndarray:
    """Basic symmetric notch reject filter around center plus/minus an offset."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    r = max(float(radius), 1.0)
    d1 = np.sqrt((y - (crow + offset_y)) ** 2 + (x - (ccol + offset_x)) ** 2)
    d2 = np.sqrt((y - (crow - offset_y)) ** 2 + (x - (ccol - offset_x)) ** 2)
    mask = np.ones(shape, dtype=np.float64)
    mask[(d1 <= r) | (d2 <= r)] = 0.0
    return mask


FILTER_LABELS = {
    "理想低通": ideal_lowpass,
    "理想高通": ideal_highpass,
    "理想带通": ideal_bandpass,
    "理想带阻": ideal_bandreject,
    "高斯低通": gaussian_lowpass,
    "高斯高通": gaussian_highpass,
    "巴特沃斯低通": butterworth_lowpass,
    "巴特沃斯高通": butterworth_highpass,
    "陷波带阻": notch_reject,
}
