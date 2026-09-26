"""Carga e inferencia local con el clasificador BERT entrenado."""

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model" / "bert_tuning" / "toxic_classifier"
REQUIRED_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def select_device(name="auto"):
    """Selecciona el acelerador solicitado o el mejor dispositivo disponible."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _validate_artifacts(model_dir):
    missing = [name for name in REQUIRED_FILES if not (model_dir / name).is_file()]
    if not any((model_dir / name).is_file() for name in WEIGHT_FILES):
        missing.append("model.safetensors (or pytorch_model.bin)")
    if missing:
        missing_files = ", ".join(missing)
        raise FileNotFoundError(
            f"El clasificador BERT está incompleto en {model_dir}. "
            f"Falta: {missing_files}."
        )


def _class_ids(config):
    """Obtiene el orden de clases guardado con el checkpoint."""
    label2id = {str(label).upper(): int(index) for label, index in config.label2id.items()}
    non_toxic_id = label2id.get("NON_TOXIC")
    toxic_id = label2id.get("TOXIC")
    if non_toxic_id is None or toxic_id is None or non_toxic_id == toxic_id:
        raise ValueError(
            "El config.json debe contener las etiquetas NON_TOXIC y TOXIC."
        )
    return non_toxic_id, toxic_id


def load_model(model_dir=MODEL_DIR, device="auto"):
    """Carga solamente los archivos locales producidos por el notebook BERT."""
    model_dir = Path(model_dir).expanduser().resolve()
    _validate_artifacts(model_dir)
    selected_device = select_device(device)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True,
        )
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"No se pudo cargar el clasificador BERT desde {model_dir}: {error}"
        ) from error

    if model.config.num_labels != 2:
        raise ValueError(
            f"El modelo debe tener 2 clases, pero config.json indica {model.config.num_labels}."
        )
    _class_ids(model.config)
    model.to(selected_device).eval()
    return model, tokenizer, selected_device


def predict_text(text, model, tokenizer, device):
    """Clasifica texto y devuelve probabilidades como [non-toxic, toxic]."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Escribe un comentario antes de analizarlo.")

    max_length = int(model.config.max_position_embeddings)
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}

    model.eval()
    with torch.inference_mode():
        raw_probabilities = torch.softmax(model(**encoded).logits, dim=-1)[0]

    non_toxic_id, toxic_id = _class_ids(model.config)
    probabilities = torch.stack(
        (raw_probabilities[non_toxic_id], raw_probabilities[toxic_id])
    ).cpu().tolist()
    prediction = 1 if probabilities[1] > probabilities[0] else 0
    return prediction, probabilities
