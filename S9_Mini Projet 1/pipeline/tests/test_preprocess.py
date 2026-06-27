import pytest
from transformers import DistilBertTokenizer

from steps.preprocess import _tokenize_batch, tokenize_data

TOKENIZER_NAME = "distilbert-base-uncased"


@pytest.fixture(scope="module")
def tokenizer() -> DistilBertTokenizer:
    return DistilBertTokenizer.from_pretrained(TOKENIZER_NAME)


class TestTokenizeBatch:
    def test_output_keys(self, tokenizer):
        enc = _tokenize_batch(["I love this film."], tokenizer, max_length=32)
        assert "input_ids" in enc
        assert "attention_mask" in enc

    def test_sequence_length_padded(self, tokenizer):
        max_len = 32
        enc = _tokenize_batch(["short"], tokenizer, max_length=max_len)
        assert len(enc["input_ids"][0]) == max_len
        assert len(enc["attention_mask"][0]) == max_len

    def test_sequence_length_truncated(self, tokenizer):
        max_len = 16
        long_text = ["word " * 300]
        enc = _tokenize_batch(long_text, tokenizer, max_length=max_len)
        assert len(enc["input_ids"][0]) == max_len

    def test_batch_size_matches_input(self, tokenizer):
        texts = ["first", "second", "third"]
        enc = _tokenize_batch(texts, tokenizer, max_length=16)
        assert len(enc["input_ids"]) == len(texts)
        assert len(enc["attention_mask"]) == len(texts)

    def test_attention_mask_values(self, tokenizer):
        enc = _tokenize_batch(["hello"], tokenizer, max_length=16)
        mask = enc["attention_mask"][0]
        assert all(v in (0, 1) for v in mask)


class TestTokenizeDataStep:
    def test_output_columns(self, sample_train_df, sample_test_df, tokenizer):
        train_tok, test_tok = tokenize_data.entrypoint(
            train_df=sample_train_df,
            test_df=sample_test_df,
            max_length=32,
        )
        for df in (train_tok, test_tok):
            assert "input_ids" in df.columns
            assert "attention_mask" in df.columns
            assert "label" in df.columns

    def test_no_extra_columns(self, sample_train_df, sample_test_df):
        train_tok, _ = tokenize_data.entrypoint(
            train_df=sample_train_df,
            test_df=sample_test_df,
            max_length=32,
        )
        assert set(train_tok.columns) == {"input_ids", "attention_mask", "label"}

    def test_row_count_preserved(self, sample_train_df, sample_test_df):
        train_tok, test_tok = tokenize_data.entrypoint(
            train_df=sample_train_df,
            test_df=sample_test_df,
            max_length=32,
        )
        assert len(train_tok) == len(sample_train_df)
        assert len(test_tok) == len(sample_test_df)
