"""Exercise the desktop widgets and worker with the saved model, offscreen."""

import os
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import torch
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from src.app import ModerationWindow


class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = ModerationWindow(device="cpu")
        self.window.show()
        self.app.processEvents()
        self.assertEqual(self.window.windowTitle(), "BERT Comments Filter")
        self.assertEqual(self.window.model.config.model_type, "bert")

    def tearDown(self):
        if self.window._worker is not None:
            self.window._worker.wait(10_000)
            self.app.processEvents()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def wait_for_result(self):
        deadline = time.monotonic() + 10
        while self.window._worker is not None and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertIsNone(self.window._worker, "Inference did not finish")

    def test_buttons_classify_clear_and_reject_empty(self):
        for text, decision in (("Thank you for your help!", "Do not ban comment"),
                               ("You are such an idiot", "Ban Comment")):
            self.window.comment.setPlainText(text)
            QTest.mouseClick(self.window.analyze_button, Qt.MouseButton.LeftButton)
            self.assertFalse(self.window.analyze_button.isEnabled())
            self.wait_for_result()
            self.assertEqual(self.window.decision.text(), decision)
            self.assertEqual(self.window.reviewed.toPlainText(), text)
            self.assertAlmostEqual(sum(b.value() for b in self.window.probability_bars), 1000, delta=1)
            self.assertTrue(all("%" in label.text() for label in self.window.probability_labels))
            self.assertTrue(self.window.analyze_button.isEnabled())
        QTest.mouseClick(self.window.clear_button, Qt.MouseButton.LeftButton)
        self.assertEqual(self.window.comment.toPlainText(), "")
        self.assertEqual(self.window.reviewed.toPlainText(), "")
        self.assertTrue(all(b.value() == 0 for b in self.window.probability_bars))
        QTest.mouseClick(self.window.analyze_button, Qt.MouseButton.LeftButton)
        self.assertEqual(self.window.decision.text(), "Write a comment to begin.")
        self.assertIsNone(self.window._worker)

    def test_worker_failure_restores_controls(self):
        self.window.comment.setPlainText("Test comment")
        with patch("src.app.predict_text", side_effect=RuntimeError("Test inference failure")):
            self.window.analyze()
            self.wait_for_result()
        self.assertEqual(self.window.decision.text(), "The comment can't be analyzed.")
        self.assertEqual(self.window.status.text(), "Test inference failure")
        self.assertTrue(self.window.analyze_button.isEnabled())
        self.assertFalse(self.window.comment.isReadOnly())

    def test_close_during_prediction_finishes_safely(self):
        self.window.comment.setPlainText("Thank you")
        self.window.analyze()
        self.window.close()
        self.assertTrue(self.window._closing)
        self.wait_for_result()
        self.assertFalse(self.window.isVisible())


if __name__ == "__main__":
    unittest.main()
