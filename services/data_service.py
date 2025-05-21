import torch


class DataService:
    """
    Provides methods to generate features/labels and perform train/test split.
    """

    @staticmethod
    def generate_features_and_labels(
            start: int,
            end: int,
            step: float,
            weight: float = 0.7,
            bias: float = 0.3,
            dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates features and labels for a linear regression model based
        on the given parameters.

        The features are generated as a range of values from `start` to `end`
        with a specified `step`.

        The labels are calculated using the linear equation: `y = weight * x + bias`.

        Parameters
        ----------
        start : int
            Starting value for the range of features
        end : int
            Ending value for the range of features
        step : float
            Step size for generation of features
        weight : float
            Weight (slope) for the linear equation
        bias : float
            Bias (intercept) for the linear equation
        dim : int
            Dimension of the features tensor. Defaults to 1.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing:

                - features: Tensor of generated features
                - labels: Tensor of calculated labels based on the linear equation
        """

        features = torch.arange(start, end, step).unsqueeze(dim=dim)
        labels = weight * features + bias  # Linear regression: y = b*x + a

        return features, labels

    @staticmethod
    def perform_train_test_split(
            features: torch.Tensor,
            labels: torch.Tensor,
            train_sample_percent: float = 0.8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Splits the features and labels into training and testing sets.
        The split is done based on the specified percentage of training samples.

        Parameters
        ----------
        features : torch.Tensor
            Tensor containing the features
        labels : torch.Tensor
            Tensor containing the labels
        train_sample_percent : float
            Percentage of data to be used for training. Defaults to 0.8 (80%).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing:

                - features_train: Tensor of training features
                - labels_train: Tensor of training labels
                - features_test: Tensor of testing features
                - labels_test: Tensor of testing labels
        """

        train_split = int(train_sample_percent * len(features))
        features_train, labels_train = features[:train_split], labels[:train_split]
        features_test, labels_test = features[train_split:], labels[train_split:]

        return features_train, labels_train, features_test, labels_test
