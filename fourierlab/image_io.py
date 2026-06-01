"""Image loading, saving, resizing, and display conversion helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


SUPPORTED_EXTENSIONS = "*.jpg *.jpeg *.png *.bmp"


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize an array to uint8 in [0, 255]."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.size == 0:
        return arr.astype(np.uint8)
    min_val = float(np.nanmin(arr))
    max_val = float(np.nanmax(arr))
    if not np.isfinite(min_val) or not np.isfinite(max_val) or max_val <= min_val:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = (arr - min_val) / (max_val - min_val) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def clip_to_uint8(image: np.ndarray) -> np.ndarray:
    """Clip an image array to uint8 display range."""
    return np.clip(np.asarray(image), 0, 255).astype(np.uint8)


def resize_max_side(image: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """Resize image so its largest side is at most max_side."""
    h, w = image.shape[:2]
    largest = max(h, w)
    if largest <= max_side:
        return image
    scale = max_side / largest
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    pil = Image.fromarray(clip_to_uint8(image))
    resample = Image.Resampling.LANCZOS
    return np.asarray(pil.resize(new_size, resample))


def load_image(path: str | Path, grayscale: bool = True, max_side: int = 1024) -> np.ndarray:
    """Load an image from disk, defaulting to grayscale for Fourier processing."""
    mode = "L" if grayscale else "RGB"
    with Image.open(path) as im:
        arr = np.asarray(im.convert(mode))
    return resize_max_side(arr, max_side=max_side).astype(np.float64)


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Save a grayscale or RGB image to disk."""
    arr = clip_to_uint8(image)
    Image.fromarray(arr).save(path)
