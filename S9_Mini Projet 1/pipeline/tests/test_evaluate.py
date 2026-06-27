import numpy as np
import pytest
import torch

from steps.evaluate import _compute_metrics, _make_tensor_dataset


class TestComputeMetrics:
    @pytest.mark.parametrize("labels, preds, expected_acc", [
        ([0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 0, 1], 1.0),          # parfait
        ([0, 0, 1, 1],       [1, 1, 0, 0],       0.0),            # inverse
        ([0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1], 0.5),  # equilibre
    ])
    def test_accuracy(self, labels, preds, expected_acc):
        m = _compute_metrics(labels, preds)
        assert m["accuracy"] == pytest.approx(expected_acc)

    def test_metrics_in_valid_range(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 2, size=100).tolist()
        preds = rng.integers(0, 2, size=100).tolist()
        m = _compute_metrics(labels, preds)
        for key, val in m.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} hors [0, 1]"

    def test_all_keys_present(self):
        m = _compute_metrics([0, 1, 0], [0, 0, 1])
        assert set(m.keys()) == {"accuracy", "f1", "precision", "recall"}

    def test_all_values_are_float(self):
        m = _compute_metrics([0, 1, 0, 1], [1, 1, 0, 0])
        for val in m.values():
            assert isinstance(val, float)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            _compute_metrics([0, 1, 0], [0, 1])


class TestMakeTensorDataset:
    def test_length_matches_dataframe(self, sample_tokenized_df):
        dataset = _make_tensor_dataset(sample_tokenized_df)
        assert len(dataset) == len(sample_tokenized_df)

    def test_item_shapes(self, sample_tokenized_df):
        dataset = _make_tensor_dataset(sample_tokenized_df)
        ids, mask, label = dataset[0]
        max_len = len(sample_tokenized_df["input_ids"].iloc[0])
        assert ids.shape == torch.Size([max_len])
        assert mask.shape == torch.Size([max_len])
        assert label.shape == torch.Size([])

    def test_dtypes(self, sample_tokenized_df):
        dataset = _make_tensor_dataset(sample_tokenized_df)
        ids, mask, label = dataset[0]
        assert ids.dtype == torch.long
        assert mask.dtype == torch.long
        assert label.dtype == torch.long
