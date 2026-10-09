"""Command-line interface for sentiment analyzer."""

import argparse
import sys
from pathlib import Path

from .model import SentimentModel, get_tokenizer, predict_sentiment


def main():
    parser = argparse.ArgumentParser(description="Sentiment Analyzer - IMDb reviews")
    subparsers = parser.add_subparsers(dest="command")

    # Train command
    train_p = subparsers.add_parser("train", help="Train a sentiment model")
    train_p.add_argument("--model", default="distilbert-base-uncased")
    train_p.add_argument("--epochs", type=int, default=3)
    train_p.add_argument("--output", default="models/sentiment_model")
    train_p.add_argument("--batch-size", type=int, default=16)
    train_p.add_argument("--lr", type=float, default=2e-5)

    # Predict command
    predict_p = subparsers.add_parser("predict", help="Predict sentiment")
    predict_p.add_argument("--model", required=True, help="Path to saved model")
    predict_p.add_argument("--text", required=True, nargs="+", help="Text to analyze")

    args = parser.parse_args()

    if args.command == "train":
        print("Loading data...")
        from .preprocessor import load_imdb_data
        train_texts, train_labels, test_texts, test_labels = load_imdb_data(num_samples=1000)

        print("Initializing model...")
        tokenizer = get_tokenizer(args.model)
        model = SentimentModel(args.model)

        print("Training...")
        from .trainer import train_model
        history = train_model(
            model,
            train_texts,
            train_labels,
            test_texts,
            test_labels,
            tokenizer,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )

        print(f"Training complete. History: {history}")
        Path(args.output).mkdir(parents=True, exist_ok=True)
        from .trainer import save_model
        save_model(model, tokenizer, args.output)
        print(f"Model saved to {args.output}")

    elif args.command == "predict":
        print("Loading model...")
        tokenizer = get_tokenizer(args.model)
        model = SentimentModel.from_pretrained(args.model)
        model.eval()

        texts = [" ".join(args.text)]
        predictions = predict_sentiment(model, texts, tokenizer)
        labels = ["positive" if p == 1 else "negative" for p in predictions]

        for text, label in zip(texts, labels):
            print(f"Text: {text[:50]}...")
            print(f"Sentiment: {label}\n")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
