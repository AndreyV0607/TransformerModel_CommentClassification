"""Entrena y guarda el BPE del notebook usando solo el split de entrenamiento."""

import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

if __package__:
    from .data import DATA_SOURCE, load_splits
    from .model import MAX_LEN, MODEL_DIR, MODEL_FILENAME, SEED, TOKENIZER_FILENAME, VOCAB_SIZE
else:
    from data import DATA_SOURCE, load_splits
    from model import MAX_LEN, MODEL_DIR, MODEL_FILENAME, SEED, TOKENIZER_FILENAME, VOCAB_SIZE


def train_tokenizer(texts):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"],
        min_frequency=2,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    tokenizer.enable_truncation(max_length=MAX_LEN)
    tokenizer.enable_padding(
        length=MAX_LEN, pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]",
    )
    return tokenizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-csv", default=DATA_SOURCE, help="CSV local o URI hf:// del notebook.")
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_path = output_dir / TOKENIZER_FILENAME
    # A newly trained BPE can assign different IDs, even with the same vocabulary size.
    if output_path.exists() or (output_dir / MODEL_FILENAME).exists():
        parser.error("Ya hay un tokenizer o modelo guardado. Usa --output-dir model/nuevo_entrenamiento.")
    splits = load_splits(args.data_csv, seed=SEED)
    texts, _ = splits["train"]
    print(f"Entrenando tokenizer con {len(texts):,} comentarios del split de entrenamiento.")
    tokenizer = train_tokenizer(texts)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    print(f"Vocabulario: {tokenizer.get_vocab_size():,} tokens")
    print(f"Tokenizer guardado en: {output_path}")


if __name__ == "__main__":
    main()
