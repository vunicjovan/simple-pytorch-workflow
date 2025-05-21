import logging
import os.path
from logging import Logger
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def configure_logging() -> Logger:
    logs_dir = "logs"
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                filename=os.path.join(logs_dir, "main.log"),
                mode="w",
            ),
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


class LinearRegressionModel(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear_layer = torch.nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


def train(
        number_of_epochs: int,
        model_to_be_trained: LinearRegressionModel,
        training_data: torch.Tensor,
        training_labels: torch.Tensor,
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        loss_function: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        logger: Logger,
) -> tuple:
    torch.manual_seed(42)

    predicted_train_labels = None
    predicted_test_labels = None

    for epoch in range(number_of_epochs):
        # Enter the training mode
        model_to_be_trained.train()

        # 1) Forward pass
        predicted_train_labels = model_to_be_trained(training_data)
        # 2) Calculate the loss
        loss = loss_function(predicted_train_labels, training_labels)
        # 3) Zero grad optimizer
        optimizer.zero_grad()
        # 4) Loss backward
        loss.backward()
        # 5) Step the optimizer
        optimizer.step()

        # Enter the evaluation mode
        model_to_be_trained.eval()

        with torch.inference_mode():
            # 1) Forward pass
            predicted_test_labels = model_to_be_trained(test_data)
            # 2) Calculate the loss
            test_loss = loss_function(predicted_test_labels, test_labels)

        logger.info(f"Epoch {epoch} | Train loss: {loss} | Test loss: {test_loss}")

    logger.info(f"Trained model's parameters: {model_to_be_trained.state_dict()}")

    return predicted_train_labels, predicted_test_labels


def save_model(model_to_be_saved: LinearRegressionModel, file_path: str, logger: Logger) -> None:
    logger.info(f"Saving model to: {file_path} ...")
    torch.save(obj=model_to_be_saved.state_dict(), f=file_path)
    logger.info(f"Model saved successfully to: {file_path}.")


def load_model(file_path: str, logger: Logger) -> LinearRegressionModel:
    logger.info(f"Loading model state from: {file_path} ...")
    loaded_model = LinearRegressionModel()
    loaded_model.load_state_dict(torch.load(f=file_path))
    logger.info(f"Model state loaded successfully from {file_path}: {loaded_model.state_dict()}.")

    return loaded_model


def main():
    # Obtain the logger instance
    logger = configure_logging()

    # Set the device-agnostic code
    device = set_device()
    logger.info(f"Device being used: {device}")

    # Create X and y (feature and labels, respectively)
    X, y = create_features_and_labels(weight=0.7, bias=0.3, start=0, end=1, step=0.02, dim=1)

    # Perform train-test split for the created data: train 80%, test 20%
    X_train, y_train, X_test, y_test = perform_train_test_split(features=X, labels=y, train_sample_percent=0.8)

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Create a linear regression model
    torch.manual_seed(seed=42)
    linear_regression_model = LinearRegressionModel()

    # Move model to the currently available device (preferably GPU - if it is available)
    linear_regression_model.to(device=device)

    # Create loss function and optimizer
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(params=linear_regression_model.parameters(), lr=0.01)

    # Train the model
    training_predictions, test_predictions = train(
        number_of_epochs=1_000,
        model_to_be_trained=linear_regression_model,
        training_data=X_train,
        training_labels=y_train,
        test_data=X_test,
        test_labels=y_test,
        loss_function=loss_fn,
        optimizer=optimizer,
        logger=logger,
    )

    # Visualize the predictions
    plot_predictions(
        train_data=X_train.detach().cpu().numpy(),
        train_labels=y_train.detach().cpu().numpy(),
        test_data=X_test.detach().cpu().numpy(),
        test_labels=y_test.detach().cpu().numpy(),
        predictions=test_predictions.detach().cpu().numpy(),
    )

    # Save the model
    model_directory = "models"
    model_file_name = "linear_regression_model.pth"
    model_file_path = os.path.join(model_directory, model_file_name)

    Path(model_directory).mkdir(exist_ok=True, parents=True)

    save_model(
        model_to_be_saved=linear_regression_model,
        file_path=model_file_path,
        logger=logger,
    )

    # Load the previously saved model
    loaded_linear_regression_model = load_model(file_path=model_file_path, logger=logger)
    loaded_linear_regression_model.to(device)

    # Evaluate the loaded model
    loaded_linear_regression_model.eval()

    with torch.inference_mode():
        loaded_linear_regression_model_predictions = loaded_linear_regression_model(X_test)

    logger.info(test_predictions == loaded_linear_regression_model_predictions)


if __name__ == "__main__":
    main()

# Run the script with: uv run main.py

# TODO: Set up Docker and docker-compose
# TODO: Write tests
# TODO: Split the code into modules
