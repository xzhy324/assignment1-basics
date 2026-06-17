import torch
from torch import Tensor
from jaxtyping import Float


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """Construct a linear transformation module.
        Args:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        raise NotImplementedError("You need to implement this function.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input.
        Args:
            x: torch.Tensor Input tensor of shape (batch_size, in_features)
        Returns:
            torch.Tensor Output tensor of shape (batch_size, out_features)
        """
        raise NotImplementedError("You need to implement this function.")

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    raise NotImplementedError("You need to implement this function.")
