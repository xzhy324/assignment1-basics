import torch
from torch import Tensor
from jaxtyping import Float
import einops


class Linear(torch.nn.Module):
    """A fully connected linear layer."""
    weight: Float[Tensor, "out_features in_features"]

    def __init__(self, in_features, out_features, device=None, dtype=None):
        """Construct a linear transformation module.
        Args:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        # initialize the weights using truncated normal distribution
        # 𝒩︀(𝜇 = 0, 𝜎2 = 2/(𝑑in+𝑑out) ) truncated at [−3𝜎, 3𝜎].
        variance = 2 / (in_features + out_features)
        std = variance**0.5
        torch.nn.init.trunc_normal_(
            self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input.
        Args:
            x: torch.Tensor Input tensor of shape (batch_size, in_features)
        Returns:
            torch.Tensor Output tensor of shape (batch_size, out_features)
        """
        return einops.einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


# glue code for `uv run pytest -k test_linear`
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
    w = Linear(d_in, d_out)
    # use use Module.load_state_dict to load the weights into the linear module
    w.load_state_dict({"weight": weights})
    y = w.forward(in_features)
    return y
