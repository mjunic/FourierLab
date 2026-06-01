"""Core FFT, spectrum, and inverse reconstruction routines."""

from __future__ import annotations

import numpy as np

from .image_io import clip_to_uint8, normalize_to_uint8


def fft2_shift(image: np.ndarray) -> np.ndarray:
    """Compute centered 2D FFT over image height and width.

    For RGB images, each channel is transformed independently.
    """
    return np.fft.fftshift(np.fft.fft2(np.asarray(image, dtype=np.float64), axes=(0, 1)), axes=(0, 1))


def magnitude_spectrum(fft_shifted: np.ndarray) -> np.ndarray:
    """Return displayable log amplitude spectrum log(abs(F)+1)."""
    spectrum = np.log1p(np.abs(fft_shifted))
    if spectrum.ndim == 3:
        return np.mean(spectrum, axis=2)
    return spectrum


def phase_spectrum(fft_shifted: np.ndarray) -> np.ndarray:
    """Return displayable phase spectrum in radians."""
    phase = np.angle(fft_shifted)
    if phase.ndim == 3:
        return np.mean(phase, axis=2)
    return phase


def apply_filter(fft_shifted: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a frequency-domain mask."""
    if fft_shifted.ndim == 3 and mask.ndim == 2:
        mask = mask[:, :, np.newaxis]
    return fft_shifted * mask


def ifft2_reconstruct(fft_shifted_filtered: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Reconstruct image from centered FFT."""
    inv = np.fft.ifft2(np.fft.ifftshift(fft_shifted_filtered, axes=(0, 1)), axes=(0, 1))
    image = np.real(inv)
    if normalize:
        if image.ndim == 3:
            return clip_to_uint8(image)
        return normalize_to_uint8(image)
    return image


def high_frequency_component(image: np.ndarray, lowpass_mask: np.ndarray) -> np.ndarray:
    """Extract high-frequency component using image minus low-pass reconstruction."""
    fft = fft2_shift(image)
    low = ifft2_reconstruct(apply_filter(fft, lowpass_mask), normalize=False)
    return np.asarray(image, dtype=np.float64) - low
