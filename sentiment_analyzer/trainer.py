"""Training utilities for sentiment analyzer."""

import torch
import numpy as np
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader


class IMDbDataset(Dataset):
    """PyTorch Dataset for IMDb data."""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Initialize the dataset.

        Args:
            texts: List of text strings.
            labels: List of integer labels.
            tokenizer: Hugging Face tokenizer.
            max_length: Maximum sequence length.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train_model(
    model,
    train_texts,
    train_labels,
    test_texts,
    test_labels,
    tokenizer,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    device: str = "cpu",
) -> dict:
    """
    Train the sentiment model.

    Args:
        model: SentimentModel instance.
        train_texts: Training text list.
        train_labels: Training label list.
        test_texts: Test text list.
        test_labels: Test label list.
        tokenizer: Hugging Face tokenizer.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        device: Device to train on.

    Returns:
        Dictionary of training metrics.
    """
    model = model.to(device)
    model.train()

    train_dataset = IMDbDataset(train_texts, train_labels, tokenizer)
    test_dataset = IMDbDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    history = {"loss": [], "accuracy": []}

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total

        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)

        # Evaluate on test set
        test_acc = evaluate(model, test_texts, test_labels, tokenizer, device)
        history[f"epoch_{epoch+1}_test_accuracy"] = test_acc

    return history


def evaluate(model, texts, labels, tokenizer, device="cpu") -> float:
    """
    Evaluate model accuracy on given data.

    Args:
        model: Trained SentimentModel.
        texts: List of text strings.
        labels: List of true labels.
        tokenizer: Hugging Face tokenizer.
        device: Device to evaluate on.

    Returns:
        Accuracy score.
    """
    model.eval()
    dataset = IMDbDataset(texts, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

            total += labels_batch.size(0)
            correct += (predictions == labels_batch).sum().item()

    return correct / total if total > 0 else 0.0


def save_model(model, tokenizer, filepath: str) -> None:
    """Save model and tokenizer."""
    torch.save(model.state_dict(), filepath)
    tokenizer.save_pretrained(filepath)


def load_model(model_name: str, model_path: str, device: str = "cpu"):
    """Load a saved model."""
    from .model import SentimentModel
    tokenizer = AutoTokenizer.from_pretrained(model_path or model_name)
    model = SentimentModel.from_pretrained(model_path or model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model