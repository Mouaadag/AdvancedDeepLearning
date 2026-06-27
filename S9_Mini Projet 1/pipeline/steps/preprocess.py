import logging
from typing import List, Tuple, Annotated

import pandas as pd
from transformers import DistilBertTokenizer
from zenml import step

logger = logging.getLogger(__name__)

TOKENIZER_NAME = "distilbert-base-uncased"


def _tokenize_batch(
    texts: List[str],
    tokenizer: DistilBertTokenizer,
    max_length: int,
) -> dict:
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )


@step
def tokenize_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_length: int = 128,
) -> Tuple[
    Annotated[pd.DataFrame, "train_tokens"],
    Annotated[pd.DataFrame, "test_tokens"],
]:
    """Tokenise les textes avec DistilBertTokenizer et ajoute input_ids / attention_mask."""
    tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_NAME)
    cols = ["input_ids", "attention_mask", "label"]

    for df in (train_df, test_df):
        enc = _tokenize_batch(df["text"].tolist(), tokenizer, max_length)
        df["input_ids"] = enc["input_ids"]
        df["attention_mask"] = enc["attention_mask"]

    logger.info("Tokenisation terminee. max_length=%d", max_length)
    return train_df[cols].copy(), test_df[cols].copy()
