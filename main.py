import logging
from logging import Logger
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def configure_logging() -> Logger:
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(__name__)


def set_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_features_and_labels(
        weight: float,
        bias: float,
        start: int,
        end: int,
        step: float,
        dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.arange(start, end, step).unsqueeze(dim=dim)
    labels = weight * features + bias  # Linear regression: y = b*x + a

    return features, labels


def perform_train_test_split(
        features: torch.Tensor,
        labels: torch.Tensor,
        train_sample_percent: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_split = int(train_sample_percent * len(features))
    features_train, labels_train = features[:train_split], labels[:train_split]
    features_test, labels_test = features[train_split:], labels[train_split:]

    return features_train, labels_train, features_test, labels_test


def plot_predictions(
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        predictions: torch.Tensor = None,
) -> None:
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, color="blue", label="Training Data", s=4)

    # Plot test data in green
    plt.scatter(test_data, test_labels, color="green", label="Test Data", s=4)

    # Plot predictions in red if provided
    if predictions is not None:
        plt.scatter(test_data, predictions, color="red", label="Predictions", s=4)

    plt.legend(prop={"size": 14})
    plt.show()


def main():
    # Set the device-agnostic code
    device = set_device()
    print(f"Device being used: {device}")

    # Create X and y (feature and labels, respectively)
    X, y = create_features_and_labels(weight=0.7, bias=0.3, start=0, end=10, step=0.02, dim=1)

    # Perform train-test split for the created data
    X_train, y_train, X_test, y_test = perform_train_test_split(features=X, labels=y, train_sample_percent=0.8)

    #
    plot_predictions(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()

# Run the script with: uv run main.py
