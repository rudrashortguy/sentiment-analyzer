"""Data preprocessing module for sentiment analyzer."""

import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def load_imdb_data(
    test_size: float = 0.2,
    random_state: int = 42,
    num_samples: int = None,
) -> tuple:
    """
    Load IMDb dataset and split into train/test sets.

    Args:
        test_size: Proportion of data for testing.
        random_state: Random seed.
        num_samples: Optional limit on samples per class for faster testing.

    Returns:
        Tuple of (train_texts, train_labels, test_texts, test_labels)
    """
    dataset = load_dataset("imdb")

    train_data = dataset["train"]
    test_data = dataset["test"]

    # Limit samples if specified
    if num_samples:
        train_data = train_data.select(range(min(num_samples, len(train_data))))
        test_data = test_data.select(range(min(num_samples, len(test_data))))

    train_texts = train_data["text"]
    train_labels = train_data["label"]
    test_texts = test_data["text"]
    test_labels = test_data["label"]

    return train_texts, train_labels, test_texts, test_labels


def preprocess_texts(
    texts: list,
    tokenizer,
    max_length: int = 512,
    truncation: bool = True,
) -> dict:
    """
    Tokenize texts for model input.

    Args:
        texts: List of text strings.
        tokenizer: Hugging Face tokenizer.
        max_length: Maximum sequence length.
        truncation: Whether to truncate long sequences.

    Returns:
        Dictionary of tokenized inputs.
    """
    return tokenizer(
        texts,
        padding=True,
        truncation=truncation,
        max_length=max_length,
        return_tensors="pt",
    )