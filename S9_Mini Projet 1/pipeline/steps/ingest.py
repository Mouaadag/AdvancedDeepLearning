import logging
import pandas as pd
from typing import Tuple, Annotated

from datasets import load_dataset
from zenml import step

logger = logging.getLogger(__name__)


def _sample_split(split, n: int, random_state: int = 42) -> pd.DataFrame:
    df = split.to_pandas()
    return df.sample(n=min(n, len(df)), random_state=random_state).reset_index(drop=True)


@step
def load_imdb(
    max_train_samples: int = 2000,
    max_test_samples: int = 500,
) -> Tuple[
    Annotated[pd.DataFrame, "train_df"],
    Annotated[pd.DataFrame, "test_df"],
]:
    """Charge le dataset IMDb depuis HuggingFace et retourne un sous-echantillon."""
    logger.info("Chargement du dataset IMDb...")
    raw = load_dataset("stanfordnlp/imdb")
    train_df = _sample_split(raw["train"], max_train_samples)
    test_df = _sample_split(raw["test"], max_test_samples)
    logger.info("Train: %d | Test: %d", len(train_df), len(test_df))
    return train_df, test_df
