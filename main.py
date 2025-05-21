import os.path
from pathlib import Path

import torch

from log.log_configuration import LogConfiguration
from models.linear_regression_model import LinearRegressionModel
from services.data_service import DataService
from services.model_service import ModelService
from services.visualization_service import VisualizationService


def main() -> None:
    # Obtain the central logger instance
    logger = LogConfiguration.configure_logging(__name__)

    # Set the device-agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device being used: {device}")

    # Create X and y (feature and labels, respectively)
    X, y = DataService.generate_features_and_labels(start=0, end=1, step=0.02)

    # Perform train-test split for the created data: train 80%, test 20%
    X_train, y_train, X_test, y_test = DataService.perform_train_test_split(features=X, labels=y)

    # Move the data to the currently available device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Create a linear regression model with manual seed
    torch.manual_seed(seed=42)
    linear_regression_model = LinearRegressionModel()

    # Move model to the currently available device
    linear_regression_model.to(device=device)

    # Create loss function and optimizer
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(params=linear_regression_model.parameters(), lr=0.01)

    # Train the model
    training_predictions, test_predictions = ModelService.train(
        number_of_epochs=1_000,
        model_to_be_trained=linear_regression_model,
        training_data=X_train,
        training_labels=y_train,
        test_data=X_test,
        test_labels=y_test,
        loss_function=loss_fn,
        optimizer=optimizer,
    )

    # Visualize the predictions
    VisualizationService.plot_predictions(
        train_data=X_train.detach().cpu().numpy(),
        train_labels=y_train.detach().cpu().numpy(),
        test_data=X_test.detach().cpu().numpy(),
        test_labels=y_test.detach().cpu().numpy(),
        predictions=test_predictions.detach().cpu().numpy(),
    )

    # Save the model
    model_directory = "stored-models"
    model_file_name = "linear_regression_model.pth"
    model_file_path = os.path.join(model_directory, model_file_name)

    Path(model_directory).mkdir(exist_ok=True, parents=True)

    ModelService.save_model(model_to_be_saved=linear_regression_model, file_path=model_file_path)

    # Load the previously saved model
    loaded_linear_regression_model = ModelService.load_model(file_path=model_file_path)
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
