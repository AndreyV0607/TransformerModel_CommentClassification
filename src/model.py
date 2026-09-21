"""Arquitectura y configuración compartidas con Model_Training.ipynb."""

from pathlib import Path

import torch
from torch import nn
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_FILENAME = "toxicity_transformer.pt"
TOKENIZER_FILENAME = "toxicity_tokenizer.json"
SEED = 42
MAX_LEN = 128
VOCAB_SIZE = 30_000


def select_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TransformerClassifier(nn.Module):
    """Mismos nombres de capas y valores por defecto que el notebook."""

    def __init__(self, vocab_size, num_classes=2, pad_id=0, max_len=128,
                 embed_dim=256, num_heads=8, num_layers=6,
                 dim_feedforward=1028, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.positional_embedding = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(input_ids) + self.positional_embedding(positions)
        x = self.transformer(x, src_key_padding_mask=(attention_mask == 0))
        return self.classifier(self.dropout(x[:, 0, :]))


def load_tokenizer(model_dir=MODEL_DIR):
    path = Path(model_dir) / TOKENIZER_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el tokenizer: {path}")
    tokenizer = Tokenizer.from_file(str(path))
    for token in ("[PAD]", "[UNK]", "[CLS]", "[SEP]"):
        if tokenizer.token_to_id(token) is None:
            raise ValueError(f"El tokenizer no contiene el token especial {token}.")
    # The saved notebook tokenizer already includes padding and truncation.
    if not tokenizer.truncation or tokenizer.truncation["max_length"] != MAX_LEN:
        raise ValueError(f"El tokenizer debe truncar a {MAX_LEN} tokens, como el notebook.")
    if not tokenizer.padding or tokenizer.padding["length"] != MAX_LEN:
        raise ValueError(f"El tokenizer debe completar a {MAX_LEN} tokens, como el notebook.")
    return tokenizer


def load_model(model_dir=MODEL_DIR, device="auto"):
    """Carga el state_dict original sin entrenar ni modificar los archivos."""
    model_dir = Path(model_dir)
    path = model_dir / MODEL_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el modelo entrenado: {path}")
    tokenizer = load_tokenizer(model_dir)
    selected_device = select_device(device)
    model = TransformerClassifier(
        vocab_size=tokenizer.get_vocab_size(),
        pad_id=tokenizer.token_to_id("[PAD]"),
    )
    weights = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    model.to(selected_device).eval()
    return model, tokenizer, selected_device


def predict_text(text, model, tokenizer, device):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Escribe un comentario antes de analizarlo.")
    encoded = tokenizer.encode(text)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([encoded.attention_mask], dtype=torch.long, device=device)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(input_ids, attention_mask), dim=1)[0]
    # argmax preserves the notebook's decision, including ties (class 0).
    return probabilities.argmax().item(), probabilities.cpu().tolist()
