import torch


class LinearRegressionModel(torch.nn.Module):
    """
    Serves as a simple linear regression model with one input and one output feature.
    The model is built upon PyTorch's `torch.nn.Module` class.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Creates a new instance of the `LinearRegressionModel` class.

        This class inherits from `torch.nn.Module` and initializes a
        linear layer with one input and one output feature.
        """

        super().__init__(*args, **kwargs)
        self.linear_layer = torch.nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        This method takes an input tensor `x` and passes it through the
        linear layer defined in the constructor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor for the model's forward pass

        Returns
        -------
        torch.Tensor
            The output tensor after passing through the linear layer
        """

        return self.linear_layer(x)
