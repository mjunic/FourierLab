"""Teaching-oriented Fourier-domain phase perturbation experiment."""

from __future__ import annotations

import numpy as np

from .fourier_core import fft2_shift, ifft2_reconstruct, magnitude_spectrum, phase_spectrum
from .image_io import normalize_to_uint8


def _phase_noise(shape: tuple[int, int], seed: int, strength: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.uniform(-np.pi, np.pi, size=shape) * float(strength)


def encrypt_image(image: np.ndarray, seed: int = 1234, strength: float = 1.0) -> dict[str, np.ndarray]:
    """Encrypt an image by adding reproducible random phase perturbation."""
    fft = fft2_shift(image)
    mag = np.abs(fft)
    phase = np.angle(fft)
    noise = _phase_noise(phase.shape, seed, strength)
    encrypted_fft = mag * np.exp(1j * (phase + noise))
    encrypted = normalize_to_uint8(ifft2_reconstruct(encrypted_fft, normalize=False))
    return {
        "encrypted": encrypted,
        "magnitude": magnitude_spectrum(fft),
        "phase": phase_spectrum(fft),
        "raw_magnitude": mag,
        "encrypted_fft": encrypted_fft,
        "noise": noise,
    }


def decrypt_from_phase(encrypted_fft: np.ndarray, raw_magnitude: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Recover image from perturbed phase using the same phase noise."""
    phase_encrypted = np.angle(encrypted_fft)
    recovered_fft = raw_magnitude * np.exp(1j * (phase_encrypted - noise))
    return normalize_to_uint8(ifft2_reconstruct(recovered_fft, normalize=False))


def decrypt_image(encrypted: np.ndarray, original_magnitude: np.ndarray, seed: int = 1234, strength: float = 1.0) -> np.ndarray:
    """Recover approximately using encrypted phase minus the same random perturbation."""
    fft = fft2_shift(encrypted)
    phase = np.angle(fft)
    noise = _phase_noise(phase.shape, seed, strength)
    mag = np.asarray(original_magnitude, dtype=np.float64)
    if mag.shape != phase.shape:
        mag = np.abs(fft)
    recovered_fft = mag * np.exp(1j * (phase - noise))
    return normalize_to_uint8(ifft2_reconstruct(recovered_fft, normalize=False))
