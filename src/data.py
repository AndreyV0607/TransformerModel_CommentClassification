"""Limpieza y split 80/10/10 idénticos para ambos entrenamientos."""

import polars as pl
from sklearn.model_selection import train_test_split

DATA_SOURCE = (
    "hf://datasets/thesofakillers/"
    "jigsaw-toxic-comment-classification-challenge/train.csv"
)
LABEL_COLUMNS = ["toxic", "severe_toxic", "insult", "identity_hate", "obscene", "threat"]


def load_splits(source=DATA_SOURCE, seed=42):
    df = pl.read_csv(source)
    df_model = (
        df.with_columns(
            pl.when(pl.any_horizontal([pl.col(name) == 1 for name in LABEL_COLUMNS]))
            .then(1).otherwise(0).alias("label")
        )
        .select(["id", "comment_text", "label"])
        .drop_nulls(["comment_text", "label"])
        .filter(pl.col("comment_text").str.strip_chars().str.len_chars() > 0)
    )
    texts = df_model["comment_text"].to_list()
    labels = df_model["label"].to_list()
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.20, random_state=seed, stratify=labels,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp,
    )
    return {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
