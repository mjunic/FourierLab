"""FourierLab application entry point."""

from __future__ import annotations

import sys
import traceback


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        from fourierlab.app import MainWindow
    except ModuleNotFoundError as exc:
        missing = exc.name or "PySide6"
        print(f"Missing dependency: {missing}")
        print("Run this in the FourierLab folder: pip install -r requirements.txt")
        return 1

    app = QApplication(sys.argv)
    app.setApplicationName("FourierLab")

    try:
        window = MainWindow()
        window.showMaximized()
        return app.exec()
    except Exception as exc:
        detail = traceback.format_exc()
        print(detail)
        QMessageBox.critical(None, "FourierLab Error", f"{exc}\n\nDetails were printed to the console.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
