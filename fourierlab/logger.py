"""Small logging helper for learning-oriented experiment notes."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QTextEdit


class ExperimentLogger:
    """Append educational logs to a QTextEdit and export them as text."""

    def __init__(self, widget: QTextEdit) -> None:
        self.widget = widget

    def append(self, title: str, body: str) -> None:
        current = self.widget.toPlainText().strip()
        block = f"[{title}]\n{body.strip()}"
        self.widget.setPlainText(f"{current}\n\n{block}".strip() if current else block)

    def clear(self) -> None:
        self.widget.clear()

    def export(self, path: str | Path) -> None:
        Path(path).write_text(self.widget.toPlainText(), encoding="utf-8")
