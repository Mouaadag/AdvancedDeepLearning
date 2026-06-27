import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from steps.ingest import _sample_split, load_imdb


def _make_fake_split(n: int) -> MagicMock:
    import numpy as np

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "text": [f"review number {i}" for i in range(n)],
            "label": rng.integers(0, 2, size=n).tolist(),
        }
    )
    split = MagicMock()
    split.to_pandas.return_value = df
    split.__len__ = MagicMock(return_value=n)
    return split


def _make_fake_dataset(n_train: int = 200, n_test: int = 100) -> dict:
    return {"train": _make_fake_split(n_train), "test": _make_fake_split(n_test)}


class TestSampleSplit:
    def test_respects_max_samples(self):
        split = _make_fake_split(100)
        result = _sample_split(split, n=50)
        assert len(result) == 50

    def test_does_not_exceed_available(self):
        split = _make_fake_split(30)
        result = _sample_split(split, n=100)
        assert len(result) == 30

    def test_output_columns(self):
        split = _make_fake_split(50)
        result = _sample_split(split, n=20)
        assert "text" in result.columns
        assert "label" in result.columns

    def test_reproducibility(self):
        split = _make_fake_split(100)
        r1 = _sample_split(split, n=40, random_state=0)
        r2 = _sample_split(split, n=40, random_state=0)
        assert list(r1.index) == list(r2.index)


class TestLoadImbdStep:
    @patch("steps.ingest.load_dataset")
    def test_returns_two_dataframes(self, mock_load):
        mock_load.return_value = _make_fake_dataset()
        train_df, test_df = load_imdb.entrypoint(
            max_train_samples=50, max_test_samples=20
        )
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)

    @patch("steps.ingest.load_dataset")
    def test_size_constraints(self, mock_load):
        mock_load.return_value = _make_fake_dataset(n_train=200, n_test=100)
        train_df, test_df = load_imdb.entrypoint(
            max_train_samples=80, max_test_samples=30
        )
        assert len(train_df) <= 80
        assert len(test_df) <= 30

    @patch("steps.ingest.load_dataset")
    def test_both_labels_present_in_train(self, mock_load):
        mock_load.return_value = _make_fake_dataset(n_train=200, n_test=100)
        train_df, _ = load_imdb.entrypoint(
            max_train_samples=100, max_test_samples=20
        )
        assert 0 in train_df["label"].values
        assert 1 in train_df["label"].values
