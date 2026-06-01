"""Image quality and spectrum statistics."""

from __future__ import annotations

import math

import numpy as np

from .filters import distance_grid


def mse(reference: np.ndarray, target: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    tar = np.asarray(target, dtype=np.float64)
    if ref.shape != tar.shape:
        return float("nan")
    return float(np.mean((ref - tar) ** 2))


def psnr(reference: np.ndarray, target: np.ndarray, max_value: float = 255.0) -> float:
    value = mse(reference, target)
    if not np.isfinite(value):
        return float("nan")
    if value == 0:
        return float("inf")
    return float(20 * math.log10(max_value / math.sqrt(value)))


def energy_stats(fft_shifted: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    mag = np.abs(fft_shifted)
    power = mag ** 2
    total = float(np.sum(power))
    shape = fft_shifted.shape[:2]
    max_radius = min(shape) / 2.0
    low_mask = distance_grid(shape) <= max_radius * 0.25
    low_energy = float(np.sum(power[low_mask]))
    high_energy = max(total - low_energy, 0.0)
    if mask is not None and power.ndim == 3 and mask.ndim == 2:
        mask = mask[:, :, np.newaxis]
    retained = float(np.sum(power * mask)) if mask is not None else total
    denom = total if total > 0 else 1.0
    return {
        "spectrum_max": float(np.max(mag)) if mag.size else 0.0,
        "spectrum_mean": float(np.mean(mag)) if mag.size else 0.0,
        "low_energy_ratio": low_energy / denom,
        "high_energy_ratio": high_energy / denom,
        "retained_energy_ratio": retained / denom,
    }
