"""Matplotlib visualisation helpers for FourierLab."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QDialog, QVBoxLayout


class Spectrum3DDialog(QDialog):
    """Dialog showing a downsampled 3D spectrum surface."""

    def __init__(self, spectrum: np.ndarray, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("3D Spectrum")
        self.resize(900, 700)

        layout = QVBoxLayout(self)
        figure = Figure(figsize=(7, 5), tight_layout=True)
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)

        ax = figure.add_subplot(111, projection="3d")
        data = downsample(spectrum, max_size=220)
        y = np.arange(data.shape[0])
        x = np.arange(data.shape[1])
        xx, yy = np.meshgrid(x, y)
        ax.plot_surface(xx, yy, data, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_title(title)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_zlabel("log|F|")
        canvas.draw()


def downsample(data: np.ndarray, max_size: int = 300) -> np.ndarray:
    """Downsample a 2D array for responsive 3D rendering."""
    arr = np.asarray(data, dtype=np.float64)
    height, width = arr.shape[:2]
    step = max(1, int(np.ceil(max(height, width) / max_size)))
    return arr[::step, ::step]
