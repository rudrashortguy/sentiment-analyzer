# Sentiment Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A transformer-based sentiment analyzer for IMDb movie reviews using Hugging Face Transformers.

## Features

- Fine-tuning BERT/DistilBERT on IMDb dataset
- Tokenization and preprocessing pipeline
- Model training with learning rate scheduling
- Evaluation metrics (accuracy, F1, precision, recall)
- Command-line interface for training and inference
- Support for multiple transformer models
- Comprehensive test suite
- GitHub Actions CI

## Installation

```bash
git clone https://github.com/rudrashortguy/sentiment-analyzer.git
cd sentiment-analyzer
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Usage

### Training a model

```bash
sentiment-analyzer train \
    --model distilbert-base-uncased \
    --epochs 3 \
    --output models/sentiment_distilbert
```

### Making predictions

```bash
sentiment-analyzer predict \
    --model models/sentiment_distilbert \
    --text "This movie was fantastic! I loved every minute of it."
```

### Python API

```python
from sentiment_analyzer.preprocessor import load_imdb_data
from sentiment_analyzer.model import SentimentModel
from sentiment_analyzer.trainer import train_model

# Load data
train_texts, train_labels, test_texts, test_labels = load_imdb_data()

# Initialize model
model = SentimentModel("distilbert-base-uncased")

# Train
trainer = train_model(model, train_texts, train_labels, epochs=3)

# Evaluate
metrics = trainer.evaluate(test_texts, test_labels)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

## Project Structure

```
sentiment-analyzer/
├── data/                  # Data storage (not tracked)
├── sentiment_analyzer/    # Source code
│   ├── __init__.py
│   ├── preprocessor.py    # Data loading and preprocessing
│   ├── model.py           # Transformer model wrapper
│   ├── trainer.py         # Training utilities
│   └── evaluator.py       # Evaluation metrics
├── tests/                 # Unit tests
├── models/                # Trained models (not tracked)
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── pyproject.toml
└── .github/workflows/tests.yml
```

## Tech Stack

- **Language**: Python 3.8+
- **NLP**: Hugging Face Transformers
- **Deep Learning**: PyTorch
- **Data Processing**: pandas, NumPy
- **Testing**: pytest
- **CI/CD**: GitHub Actions

## Roadmap

- [x] BERT/DistilBERT fine-tuning
- [x] IMDb dataset loading
- [x] Training and evaluation pipeline
- [ ] Add RoBERTa and XLM-RoBERTa support
- [ ] Aspect-based sentiment analysis
- [ ] Multi-language support
- [ ] REST API endpoint

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.