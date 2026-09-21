"""Ventana de escritorio para clasificar comentarios con el modelo guardado."""

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

if __package__:
    from .model import MODEL_DIR, load_model, predict_text
else:
    from model import MODEL_DIR, load_model, predict_text


class PredictionThread(QThread):
    """Ejecuta la inferencia mientras la ventana sigue respondiendo."""

    result = Signal(str, int, object)
    error = Signal(str)

    def __init__(self, text, model, tokenizer, device, parent=None):
        super().__init__(parent)
        self.text, self.model, self.tokenizer, self.device = text, model, tokenizer, device

    def run(self):
        try:
            prediction, probabilities = predict_text(
                self.text, self.model, self.tokenizer, self.device,
            )
            self.result.emit(self.text, prediction, probabilities)
        except Exception as error:
            self.error.emit(str(error))


class ModerationWindow(QWidget):
    def __init__(self, model_dir=MODEL_DIR, device="auto"):
        super().__init__()
        self.model, self.tokenizer, self.device = load_model(model_dir, device)
        self._worker = None
        self._closing = False
        self.setWindowTitle("Comments Filter")
        self.resize(760, 680)
        self.setMinimumSize(540, 560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        title = QLabel("Comments Filter")
        title.setFont(QFont("", 22, QFont.Weight.Bold))
        layout.addWidget(title)
        layout.addWidget(QLabel("Write a comment in english to see if the AI model thinks it's toxic or not."))
        layout.addWidget(QLabel("Your comment"))
        self.comment = QPlainTextEdit()
        self.comment.setPlaceholderText("Thank you for your help!")
        self.comment.setAccessibleName("Your comment")
        layout.addWidget(self.comment, 1)

        buttons = QHBoxLayout()
        self.analyze_button = QPushButton("Check comment")
        self.analyze_button.setDefault(True)
        self.analyze_button.clicked.connect(self.analyze)
        self.clear_button = QPushButton("Clean")
        self.clear_button.clicked.connect(self.clear)
        buttons.addWidget(self.analyze_button)
        buttons.addWidget(self.clear_button)
        layout.addLayout(buttons)

        hint = QLabel("Check for comments like the ones were use to train the model. "
                      "Would be analize a maximum of 128 tokens, so if the comment is longer than that, it would just analize the first 128 tokens.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.decision = QLabel("Waiting for a comment")
        self.decision.setTextFormat(Qt.TextFormat.PlainText)
        self.decision.setFont(QFont("", 18, QFont.Weight.Bold))
        self.decision.setWordWrap(True)
        layout.addWidget(self.decision)
        layout.addWidget(QLabel("Comment analyzed"))
        self.reviewed = QPlainTextEdit()
        self.reviewed.setReadOnly(True)
        self.reviewed.setAccessibleName("Comment analyzed")
        layout.addWidget(self.reviewed, 1)

        probabilities_box = QGroupBox("Probabilities of each class")
        probabilities_layout = QVBoxLayout(probabilities_box)
        self.probability_bars = []
        self.probability_labels = []
        for label in ("No Toxic", "Toxic"):
            probability_label = QLabel(f"{label}: —")
            probabilities_layout.addWidget(probability_label)
            self.probability_labels.append(probability_label)
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setAccessibleName(label)
            bar.setValue(0)
            bar.setTextVisible(False)
            probabilities_layout.addWidget(bar)
            self.probability_bars.append(bar)
        layout.addWidget(probabilities_box)
        self.status = QLabel("Local model ready.")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)

    def _reset_result(self):
        self.decision.setStyleSheet("")
        self.reviewed.clear()
        for label, bar, probability_label in zip(
            ("No Toxic", "Toxic"), self.probability_bars, self.probability_labels,
        ):
            bar.setValue(0)
            probability_label.setText(f"{label}: —")

    def clear(self):
        if self._worker is not None:
            return
        self.comment.clear()
        self._reset_result()
        self.decision.setText("Waiting for a comment")
        self.status.setText("Local model ready.")
        self.comment.setFocus()

    def analyze(self):
        if self._worker is not None:
            return
        text = self.comment.toPlainText()
        self._reset_result()
        if not text.strip():
            self.decision.setText("Write a comment to begin.")
            self.comment.setFocus()
            return
        self.decision.setText("Analyzing comment...")
        self.status.setText("The model is analyzing the comment in this moment.")
        self.analyze_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.comment.setReadOnly(True)
        self._worker = PredictionThread(text, self.model, self.tokenizer, self.device, self)
        self._worker.result.connect(self._show_result)
        self._worker.error.connect(self._show_error)
        self._worker.finished.connect(self._finish_prediction)
        self._worker.start()

    def _show_result(self, text, prediction, probabilities):
        self.reviewed.setPlainText(text)
        self.decision.setText("Ban Comment" if prediction == 1 else "Do not ban comment")
        self.decision.setStyleSheet(f"color: {'#d93838' if prediction == 1 else '#168447'};")
        for label, bar, probability_label, probability in zip(
            ("No Toxic", "Toxic"), self.probability_bars, self.probability_labels, probabilities,
        ):
            bar.setValue(round(probability * 1000))
            probability_label.setText(f"{label}: {probability:.1%}")
        self.status.setText("Analisis Complete.")

    def _show_error(self, message):
        self.decision.setText("The comment can't be analyzed.")
        self.status.setText(message)

    def _finish_prediction(self):
        self._worker.deleteLater()
        self._worker = None
        self.analyze_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.comment.setReadOnly(False)
        if self._closing:
            self.close()

    def closeEvent(self, event):
        # Keep the worker alive until inference finishes, then close safely.
        if self._worker is not None:
            self._closing = True
            self.status.setText("Finishing the analysis before closing the app...")
            event.ignore()
        else:
            event.accept()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setApplicationName("Filtro de comentarios")
    try:
        window = ModerationWindow(args.model_dir, args.device)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        QMessageBox.critical(None, "The model can't start", str(error))
        return 1
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
