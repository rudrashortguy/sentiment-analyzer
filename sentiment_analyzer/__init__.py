"""Sentiment Analyzer - Transformer-based IMDb sentiment."""

from .preprocessor import load_imdb_data, preprocess_texts
from .model import SentimentModel
from .trainer import train_model, save_model, load_model

__version__ = "1.0.0"
__all__ = [
    "load_imdb_data",
    "preprocess_texts",
    "SentimentModel",
    "train_model",
    "save_model",
    "load_model",
]
