import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_train_df() -> pd.DataFrame:
    texts = [
        "This movie is absolutely fantastic! I loved every minute.",
        "Terrible film. Waste of time and money.",
        "Great acting, beautiful cinematography, highly recommend.",
        "Boring plot, bad acting. Would not watch again.",
        "A masterpiece of modern cinema.",
        "Completely predictable and utterly forgettable.",
        "Stunning visuals and a gripping storyline.",
        "One of the worst films I have ever seen.",
        "The performances were outstanding across the board.",
        "Dull and lifeless. Skip this one.",
    ] * 5
    labels = ([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 5)
    return pd.DataFrame({"text": texts, "label": labels})


@pytest.fixture
def sample_test_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [
                "Excellent movie, very touching.",
                "Disappointing sequel.",
                "Outstanding performance by the lead actor.",
            ],
            "label": [1, 0, 1],
        }
    )


@pytest.fixture
def sample_tokenized_df() -> pd.DataFrame:
    n = 30
    max_len = 32
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "input_ids": [rng.integers(0, 30522, size=max_len).tolist() for _ in range(n)],
            "attention_mask": [[1] * max_len for _ in range(n)],
            "label": rng.integers(0, 2, size=n).tolist(),
        }
    )
