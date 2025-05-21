import logging

import torch

from models.linear_regression_model import LinearRegressionModel


class ModelService:
    """
    Provides services for training, saving and loading of models.
    """

    logger = logging.getLogger(__name__)

    @classmethod
    def train(
            cls,
            number_of_epochs: int,
            model_to_be_trained: LinearRegressionModel,
            training_data: torch.Tensor,
            training_labels: torch.Tensor,
            test_data: torch.Tensor,
            test_labels: torch.Tensor,
            loss_function: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            random_seed: int = 42,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Trains the model using the provided training data and labels,
        and evaluates it using the provided test data and labels.

        The training process involves a forward pass, loss calculation,
        backward pass and optimizer step for each epoch.

        The evaluation process involves a forward pass and loss calculation
        for the provided test data.

        Parameters
        ----------
        number_of_epochs : int
            The number of epochs (i.e. iterations) for training process of the given model
        model_to_be_trained : LinearRegressionModel
            The model to be put through training process
        training_data : torch.Tensor
            The training data to be used for training the model
        training_labels : torch.Tensor
            The training labels to be used for training the model
        test_data : torch.Tensor
            The test data to be used for evaluating the model
        test_labels : torch.Tensor
            The test labels to be used for evaluating the model
        loss_function : torch.nn.Module
            The loss function to be used for training the model
        optimizer : torch.optim.Optimizer
            The optimizer to be used for training the model
        random_seed : int
            The random seed to be used for reproducibility of the training process.
            Defaults to 42.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing:

            - predicted_train_labels: Tensor of predicted labels for training data
            - predicted_test_labels: Tensor of predicted labels for test data
        """

        torch.manual_seed(seed=random_seed)

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

            cls.logger.info(f"Epoch {epoch} | Train loss: {loss} | Test loss: {test_loss}")

        cls.logger.info(f"Trained model's parameters: {model_to_be_trained.state_dict()}")

        return predicted_train_labels, predicted_test_labels

    @classmethod
    def save_model(cls, model_to_be_saved: LinearRegressionModel, file_path: str) -> None:
        """
        Saves the model to the specified file path.
        The model's state dictionary is saved, containing all the parameters of the model.
        This allows for easy loading of the model later on.

        Parameters
        ----------
        model_to_be_saved : LinearRegressionModel
            The model to be saved
        file_path : str
            The file path where the model will be saved.
            It should include the file path with extension (.pth).
        """

        cls.logger.info(f"Saving model to: {file_path} ...")
        torch.save(obj=model_to_be_saved.state_dict(), f=file_path)
        cls.logger.info(f"Model saved successfully to: {file_path}.")

    @classmethod
    def load_model(cls, file_path: str) -> LinearRegressionModel:
        """
        Loads the model from the specified file path.
        The model's state dictionary is loaded, containing all the parameters of the model.

        Parameters
        ----------
        file_path : str
            The file path from which the model will be loaded.
            It should include the file path with extension (.pth).

        Returns
        -------
        LinearRegressionModel
            The loaded model with the state dictionary restored.
            This allows for easy use of the model after loading.
        """

        cls.logger.info(f"Loading model state from: {file_path} ...")
        loaded_model = LinearRegressionModel()
        loaded_model.load_state_dict(torch.load(f=file_path))
        cls.logger.info(f"Model state loaded successfully from {file_path}: {loaded_model.state_dict()}.")

        return loaded_model
