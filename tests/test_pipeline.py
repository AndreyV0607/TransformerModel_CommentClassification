"""Integration checks; never write to the original model/ or notebooks/."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from src.data import LABEL_COLUMNS, load_splits
from src.model import (
    MODEL_DIR, MODEL_FILENAME, PROJECT_ROOT, TOKENIZER_FILENAME,
    load_model, load_tokenizer, predict_text,
)
from src.train_model import main as train_model_main
from src.train_tokenizer import main as train_tokenizer_main


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model, cls.tokenizer, cls.device = load_model(device="cpu")

    def test_saved_model_matches_notebook(self):
        notebook = json.loads((PROJECT_ROOT / "notebooks/Model_Training.ipynb").read_text())
        scope = {"torch": torch, "nn": nn}
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            if source.startswith("class TransformerClassifier") or source.startswith("def predict_text("):
                exec(source, scope)
        reference = scope["TransformerClassifier"](
            vocab_size=self.tokenizer.get_vocab_size(), num_classes=2,
            pad_id=self.tokenizer.token_to_id("[PAD]"),
        )
        reference.load_state_dict(self.model.state_dict())
        for text in ("Thank you for your help!", "You are such an idiot", "🦄 " * 300):
            with self.subTest(text=text[:40]):
                expected_class, expected_probs = scope["predict_text"](
                    text, reference, self.tokenizer, self.device,
                )
                actual_class, actual_probs = predict_text(text, self.model, self.tokenizer, self.device)
                self.assertEqual(actual_class, expected_class)
                torch.testing.assert_close(torch.tensor(actual_probs), torch.tensor(expected_probs))
        self.assertEqual(predict_text("You are such an idiot", self.model, self.tokenizer, self.device)[0], 1)
        self.assertEqual(predict_text("Thank you for your help!", self.model, self.tokenizer, self.device)[0], 0)

    def test_empty_comment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Escribe un comentario"):
            predict_text(" \n\t", self.model, self.tokenizer, self.device)

    def test_existing_artifacts_are_not_overwritten(self):
        for main in (train_model_main, train_tokenizer_main):
            with self.subTest(script=main.__module__), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(["--output-dir", str(MODEL_DIR), "--data-csv", "missing.csv"])
                self.assertEqual(error.exception.code, 2)

    def test_missing_model_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "modelo entrenado"):
                load_model(directory, device="cpu")

    def test_train_save_reload_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "comments.csv"
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "comment_text", *LABEL_COLUMNS])
                writer.writeheader()
                for index in range(40):
                    label = index % 2
                    row = {"id": str(index), "comment_text":
                           f"{'stupid idiot' if label else 'thank you friend'} comment {index}"}
                    row.update({name: int(label and name == "insult") for name in LABEL_COLUMNS})
                    writer.writerow(row)
                writer.writerow({"id": "empty", "comment_text": "   ", **dict.fromkeys(LABEL_COLUMNS, 0)})
            splits = load_splits(str(csv_path))
            self.assertEqual([len(splits[key][0]) for key in ("train", "val", "test")], [32, 4, 4])
            self.assertEqual(set(splits["train"][1]), {0, 1})
            self.assertFalse(set(splits["train"][0]) & set(splits["test"][0]))
            output_dir = root / "model"
            args = ["--data-csv", str(csv_path), "--output-dir", str(output_dir)]
            train_tokenizer_main(args)
            tokenizer = load_tokenizer(output_dir)
            encoded = tokenizer.encode("🦄")
            self.assertEqual(len(encoded.ids), 128)
            self.assertEqual(encoded.ids[:3], [tokenizer.token_to_id(t) for t in ("[CLS]", "[UNK]", "[SEP]")])
            self.assertEqual(encoded.attention_mask[:4], [1, 1, 1, 0])
            self.assertEqual(len(tokenizer.encode("comment " * 500).ids), 128)
            original_tokenizer = (output_dir / TOKENIZER_FILENAME).read_bytes()
            train_model_main(args + ["--epochs", "1", "--batch-size", "16", "--device", "cpu"])
            self.assertTrue((output_dir / MODEL_FILENAME).is_file())
            self.assertEqual(original_tokenizer, (output_dir / TOKENIZER_FILENAME).read_bytes())
            model, tokenizer, device = load_model(output_dir, device="cpu")
            prediction, probabilities = predict_text("thank you", model, tokenizer, device)
            self.assertIn(prediction, [0, 1])
            self.assertAlmostEqual(sum(probabilities), 1, places=6)
            history = json.loads((output_dir / "training_history.json").read_text())
            self.assertEqual(len(history["train_loss"]), 1)
            metrics = json.loads((output_dir / "test_metrics.json").read_text())
            self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), 4)


if __name__ == "__main__":
    unittest.main()
