"""Hybrid image experiment helpers."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .fourier_core import apply_filter, fft2_shift, ifft2_reconstruct
from .filters import ideal_highpass, ideal_lowpass
from .image_io import normalize_to_uint8


def resize_like(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape
    pil = Image.fromarray(normalize_to_uint8(image))
    return np.asarray(pil.resize((target_w, target_h), Image.Resampling.LANCZOS), dtype=np.float64)


def make_hybrid(
    image_a: np.ndarray,
    image_b: np.ndarray,
    r_low: float,
    r_high: float,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse low frequencies from A and high frequencies from B."""
    a = np.asarray(image_a, dtype=np.float64)
    b = resize_like(image_b, a.shape[:2])
    low = ifft2_reconstruct(apply_filter(fft2_shift(a), ideal_lowpass(a.shape[:2], r_low)), normalize=False)
    high = ifft2_reconstruct(apply_filter(fft2_shift(b), ideal_highpass(b.shape[:2], r_high)), normalize=False)
    fused = normalize_to_uint8(alpha * low + beta * high)
    return normalize_to_uint8(low), normalize_to_uint8(high + 127.0), fused
