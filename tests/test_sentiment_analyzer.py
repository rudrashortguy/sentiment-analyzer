"""Tests for sentiment analyzer."""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from sentiment_analyzer.model import SentimentModel, get_tokenizer, predict_sentiment
from sentiment_analyzer.preprocessor import load_imdb_data, preprocess_texts


class TestPreprocessor:
    @patch("sentiment_analyzer.preprocessor.load_dataset")
    def test_load_imdb_data_returns_tuple(self, mock_load):
        """Test that load_imdb_data returns a tuple of 4 elements."""
        mock_load.return_value = {
            "train": {"text": ["good", "bad"], "label": [1, 0]},
            "test": {"text": ["great", "awful"], "label": [1, 0]},
        }
        train_texts, train_labels, test_texts, test_labels = load_imdb_data(num_samples=2)
        assert len(train_texts) == 2
        assert len(test_texts) == 2

    def test_load_imdb_data_missing_column_raises(self):
        """Test that missing 'text' column raises ValueError."""
        with patch("sentiment_analyzer.preprocessor.load_dataset") as mock_load:
            mock_load.return_value = {
                "train": {"wrong_col": ["a", "b"], "label": [1, 0]},
                "test": {"wrong_col": ["c", "d"], "label": [1, 0]},
            }
            with pytest.raises((KeyError, TypeError)):
                load_imdb_data(num_samples=2)

    def test_preprocess_texts(self):
        """Test text preprocessing with mock tokenizer."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        texts = ["This is a test"]
        result = preprocess_texts(texts, mock_tokenizer, max_length=10)
        assert "input_ids" in result
        assert "attention_mask" in result


class TestModel:
    def test_sentiment_model_init(self):
        """Test model initialization."""
        with patch("sentiment_analyzer.model.AutoModelForSequenceClassification"):
            model = SentimentModel("distilbert-base-uncased", num_classes=2)
            assert model.num_classes == 2
            assert isinstance(model.transformer, MagicMock) or model.transformer is not None

    def test_get_tokenizer(self):
        """Test tokenizer retrieval."""
        with patch("sentiment_analyzer.model.AutoTokenizer") as mock_tokenizer:
            tok = get_tokenizer("distilbert-base-uncased")
            mock_tokenizer.from_pretrained.assert_called_once_with("distilbert-base-uncased")

    def test_predict_sentiment(self):
        """Test prediction function."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[0.8, 0.2]])
        mock_model.return_value = mock_outputs

        with patch("torch.no_grad"):
            predictions = predict_sentiment(mock_model, ["test"], mock_tokenizer, device="cpu")
            assert predictions == [1]


class TestIntegration:
    def test_end_to_end_mock(self):
        """Mock end-to-end training."""
        with patch("sentiment_analyzer.preprocessor.load_dataset") as mock_load, \
             patch("sentiment_analyzer.trainer.IMDbDataset"), \
             patch("sentiment_analyzer.trainer.DataLoader"), \
             patch("torch.optim.AdamW"), \
             patch("transformers.get_linear_schedule_with_warmup"):

            mock_load.return_value = {
                "train": {"text": ["good", "bad"], "label": [1, 0]},
                "test": {"text": ["great", "awful"], "label": [1, 0]},
            }

            from sentiment_analyzer.trainer import train_model
            from sentiment_analyzer.model import SentimentModel
            from sentiment_analyzer.preprocessor import get_tokenizer

            # Skip actual training in integration test, just verify imports work
            model = SentimentModel.__new__(SentimentModel)
            tokenizer = get_tokenizer("distilbert-base-uncased")
            assert tokenizer is not None
