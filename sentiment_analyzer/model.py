"""Model module for sentiment analyzer."""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SentimentModel(nn.Module):
    """Sentiment classification model using transformer backbone."""

    def __init__(self, model_name: str = "distilbert-base-uncased", num_classes: int = 2):
        """
        Initialize the sentiment model.

        Args:
            model_name: Name or path of the transformer model.
            num_classes: Number of output classes.
        """
        super().__init__()
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass through the model.

        Args:
            input_ids: Token input IDs.
            attention_mask: Attention mask.
            labels: Optional ground truth labels.

        Returns:
            Loss (if labels provided) or logits.
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs


def get_tokenizer(model_name: str = "distilbert-base-uncased") -> AutoTokenizer:
    """
    Get tokenizer for the specified model.

    Args:
        model_name: Name of the transformer model.

    Returns:
        Hugging Face tokenizer.
    """
    return AutoTokenizer.from_pretrained(model_name)


def predict_sentiment(
    model: SentimentModel,
    texts: list,
    tokenizer: AutoTokenizer,
    device: str = "cpu",
) -> list:
    """
    Predict sentiment for a list of texts.

    Args:
        model: Trained SentimentModel.
        texts: List of text strings.
        tokenizer: Hugging Face tokenizer.
        device: Device to run inference on ('cpu' or 'cuda').

    Returns:
        List of predicted labels (0=negative, 1=positive).
    """
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions.cpu().tolist()