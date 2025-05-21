import matplotlib.pyplot as plt
import torch


class VisualizationService:
    """
    Provides services for visualizing data and model predictions.
    """

    @staticmethod
    def plot_predictions(
            train_data: torch.Tensor,
            train_labels: torch.Tensor,
            test_data: torch.Tensor,
            test_labels: torch.Tensor,
            predictions: torch.Tensor = None,
    ) -> None:
        """
        Plots the training data, test data and predictions (if provided)
        on a 2D scatter plot, using different colors for each data set.

        Parameters
        ----------
        train_data : torch.Tensor
            Training data (features)
        train_labels : torch.Tensor
            Training labels (targets)
        test_data : torch.Tensor
            Test data (features)
        test_labels : torch.Tensor
            Test labels (targets)
        predictions : torch.Tensor
            Predictions made by the model on the test data (optional)
        """

        plt.figure(figsize=(10, 7))

        # Plot training data in blue
        plt.scatter(train_data, train_labels, color="blue", label="Training Data", s=4)

        # Plot test data in green
        plt.scatter(test_data, test_labels, color="green", label="Test Data", s=4)

        # Plot predictions in red (if provided)
        if predictions is not None:
            plt.scatter(test_data, predictions, color="red", label="Predictions", s=4)

        plt.legend(prop={"size": 14})
        plt.show()
