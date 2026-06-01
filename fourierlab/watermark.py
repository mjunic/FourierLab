"""Frequency-domain text watermark embedding and extraction."""

from __future__ import annotations

import numpy as np

from .filters import distance_grid
from .fourier_core import fft2_shift, ifft2_reconstruct
from .image_io import clip_to_uint8


def text_to_bits(text: str) -> list[int]:
    data = text.encode("utf-8")
    bits: list[int] = []
    for byte in data:
        bits.extend((byte >> shift) & 1 for shift in range(7, -1, -1))
    return bits


def bits_to_text(bits: list[int]) -> tuple[str, bool]:
    usable = len(bits) - (len(bits) % 8)
    data = bytearray()
    for idx in range(0, usable, 8):
        value = 0
        for bit in bits[idx : idx + 8]:
            value = (value << 1) | int(bit)
        data.append(value)
    try:
        return data.decode("utf-8"), True
    except UnicodeDecodeError:
        return data.hex(), False


def candidate_points(shape: tuple[int, int], r1: int, r2: int) -> list[tuple[int, int]]:
    """Return one point from each conjugate pair inside a middle-frequency ring."""
    height, width = shape
    cy, cx = height // 2, width // 2
    dist = distance_grid(shape)
    points: list[tuple[int, int]] = []
    for y in range(height):
        for x in range(width):
            if not (r1 <= dist[y, x] <= r2):
                continue
            sy, sx = (2 * cy - y) % height, (2 * cx - x) % width
            if (y, x) == (cy, cx):
                continue
            if (y, x) <= (sy, sx):
                points.append((y, x))
    return points


def select_points(shape: tuple[int, int], r1: int, r2: int, seed: int, count: int) -> list[tuple[int, int]]:
    points = candidate_points(shape, r1, r2)
    if count > len(points):
        raise ValueError("水印内容过长，请缩短文字或扩大中频嵌入范围。")
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(points), size=count, replace=False)
    return [points[int(i)] for i in indices]


def _to_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 3:
        return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return arr


def _set_symmetric_magnitude(fft: np.ndarray, y: int, x: int, magnitude: float) -> None:
    height, width = fft.shape[:2]
    cy, cx = height // 2, width // 2
    sy, sx = (2 * cy - y) % height, (2 * cx - x) % width
    phase = np.angle(fft[y, x])
    fft[y, x] = magnitude * np.exp(1j * phase)
    fft[sy, sx] = np.conj(fft[y, x])


def embed_text_watermark(
    image: np.ndarray,
    text: str,
    seed: int,
    delta: float,
    r1: int,
    r2: int,
    repeat: int = 9,
) -> dict[str, np.ndarray | int | float | str | bool]:
    if not text:
        raise ValueError("请输入水印文字。")
    bits = text_to_bits(text)
    repeat = max(1, int(repeat))
    embedded_bits = [bit for bit in bits for _ in range(repeat)]
    gray = _to_gray(image)
    fft = fft2_shift(gray)
    points = select_points(gray.shape[:2], r1, r2, seed, len(embedded_bits))
    marked_fft = fft.copy()
    delta = max(float(delta), 1e-6)
    for bit, (y, x) in zip(embedded_bits, points):
        amp = np.abs(marked_fft[y, x])
        q = np.floor(amp / delta)
        new_amp = q * delta + (0.25 * delta if bit == 0 else 0.75 * delta)
        _set_symmetric_magnitude(marked_fft, y, x, new_amp)
    watermarked = ifft2_reconstruct(marked_fft, normalize=False)
    mask = np.zeros(gray.shape[:2], dtype=np.float64)
    for y, x in points:
        height, width = gray.shape[:2]
        cy, cx = height // 2, width // 2
        sy, sx = (2 * cy - y) % height, (2 * cx - x) % width
        mask[y, x] = 255
        mask[sy, sx] = 255
    return {
        "watermarked": clip_to_uint8(watermarked),
        "mask": mask,
        "bit_length": len(bits),
        "embedded_bit_length": len(embedded_bits),
        "repeat": repeat,
        "available_points": len(candidate_points(gray.shape[:2], r1, r2)),
        "fft": marked_fft,
    }


def extract_text_watermark(
    image: np.ndarray,
    seed: int,
    delta: float,
    r1: int,
    r2: int,
    bit_length: int,
    repeat: int = 9,
) -> dict[str, str | bool | list[int]]:
    gray = _to_gray(image)
    fft = fft2_shift(gray)
    repeat = max(1, int(repeat))
    points = select_points(gray.shape[:2], r1, r2, seed, int(bit_length) * repeat)
    delta = max(float(delta), 1e-6)
    bits: list[int] = []
    for y, x in points:
        amp = np.abs(fft[y, x])
        rem = amp % delta
        bits.append(1 if rem >= 0.5 * delta else 0)
    voted_bits = []
    for idx in range(0, len(bits), repeat):
        group = bits[idx : idx + repeat]
        voted_bits.append(1 if sum(group) >= (len(group) / 2.0) else 0)
    text, ok = bits_to_text(voted_bits)
    return {"text": text, "ok": ok, "bits": voted_bits}


def extract_text_from_fft(
    fft_shifted: np.ndarray,
    seed: int,
    delta: float,
    r1: int,
    r2: int,
    bit_length: int,
    repeat: int = 9,
) -> dict[str, str | bool | list[int]]:
    """Extract watermark bits directly from the preserved complex spectrum."""
    repeat = max(1, int(repeat))
    points = select_points(fft_shifted.shape[:2], r1, r2, seed, int(bit_length) * repeat)
    delta = max(float(delta), 1e-6)
    bits: list[int] = []
    for y, x in points:
        amp = np.abs(fft_shifted[y, x])
        rem = amp % delta
        bits.append(1 if rem >= 0.5 * delta else 0)
    voted_bits = []
    for idx in range(0, len(bits), repeat):
        group = bits[idx : idx + repeat]
        voted_bits.append(1 if sum(group) >= (len(group) / 2.0) else 0)
    text, ok = bits_to_text(voted_bits)
    return {"text": text, "ok": ok, "bits": voted_bits}
