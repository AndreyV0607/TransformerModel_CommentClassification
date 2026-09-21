"""Entrena el Transformer del notebook y guarda sus pesos finales."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

if __package__:
    from .data import DATA_SOURCE, load_splits
    from .model import MODEL_DIR, MODEL_FILENAME, SEED, TransformerClassifier, load_tokenizer, select_device
else:
    from data import DATA_SOURCE, load_splits
    from model import MODEL_DIR, MODEL_FILENAME, SEED, TransformerClassifier, load_tokenizer, select_device


class ToxicityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts, self.labels, self.tokenizer = texts, labels, tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer.encode(self.texts[index])
        return {
            "input_ids": torch.tensor(encoded.ids, dtype=torch.long),
            "attention_mask": torch.tensor(encoded.attention_mask, dtype=torch.long),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
        }


def train_one_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss, correct_predictions, total_examples = 0.0, 0, 0
    progress_bar = tqdm(data_loader, desc="Training")
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        correct_predictions += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(data_loader), correct_predictions / total_examples


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss, correct_predictions, total_examples = 0.0, 0, 0
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            total_loss += loss.item()
            correct_predictions += (predictions == labels).sum().item()
            total_examples += labels.size(0)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return (total_loss / len(data_loader), correct_predictions / total_examples,
            all_predictions, all_labels)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-csv", default=DATA_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR,
                        help="Carpeta que contiene el tokenizer y recibirá los pesos.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs y --batch-size deben ser mayores que cero.")
    output_dir = args.output_dir.resolve()
    output_path = output_dir / MODEL_FILENAME
    if output_path.exists():
        parser.error("Ya existe un modelo en esa carpeta. Usa --output-dir model/nuevo_entrenamiento.")
    try:
        tokenizer = load_tokenizer(output_dir)
    except (FileNotFoundError, ValueError) as error:
        parser.error(f"{error}\nEjecuta primero train_tokenizer.py con el mismo --output-dir.")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = select_device(args.device)
    print(f"Using device: {device}")
    splits = load_splits(args.data_csv, seed=SEED)
    loaders = {
        name: DataLoader(ToxicityDataset(texts, labels, tokenizer),
                         batch_size=args.batch_size, shuffle=(name == "train"))
        for name, (texts, labels) in splits.items()
    }
    for name, (texts, _) in splits.items():
        print(f"{name}: {len(texts):,} comentarios")
    model = TransformerClassifier(
        vocab_size=tokenizer.get_vocab_size(), pad_id=tokenizer.token_to_id("[PAD]"),
    ).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    class_counts = np.bincount(splits["train"][1], minlength=2)
    if (class_counts == 0).any():
        parser.error("El split de entrenamiento debe contener ambas clases.")
    class_weights = len(splits["train"][1]) / (len(class_counts) * class_counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    history = {key: [] for key in ("train_loss", "train_accuracy", "val_loss", "val_accuracy")}
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}\n" + "-" * 50)
        train_loss, train_accuracy = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device,
        )
        val_loss, val_accuracy, _, _ = evaluate(model, loaders["val"], criterion, device)
        for key, value in zip(history, (train_loss, train_accuracy, val_loss, val_accuracy)):
            history[key].append(value)
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")

    # Plain state_dict, with exactly the same layer names as the notebook.
    torch.save(model.state_dict(), output_path)
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"Modelo final guardado en: {output_path}")
    test_loss, test_accuracy, predictions, labels = evaluate(model, loaders["test"], criterion, device)
    report_args = dict(labels=[0, 1], target_names=["Non-toxic", "Toxic"], zero_division=0)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")
    print(classification_report(labels, predictions, digits=4, **report_args))
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    print(matrix)
    metrics = {
        "test_loss": test_loss, "test_accuracy": test_accuracy,
        "classification_report": classification_report(labels, predictions, output_dict=True, **report_args),
        "confusion_matrix": matrix.tolist(),
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
