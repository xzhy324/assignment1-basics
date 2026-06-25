import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def softmax(x: Tensor, dim: int) -> Tensor:
    """use softmax to normal the dim'th dimension of tensor x
    Args:
        x: input tensor
        dim: the dimension to apply softmax
    Returns:
        Tensor: same shape of x, but the dim'th dimension is normalized by softmax
    """
    # use keepdim-True to apply broadcast in later operations
    # examples:
    #   x = torch.tensor(
    #       [[1, 2, 3],
    #       [4, 5, 6]]
    #   )
    #   x.amax(dim=1, keepdim=True) -> tensor([[3], [6]]) shape: (2, 1)
    #   x.amax(dim=1, keepdim=False) -> tensor([3, 6]) shape: (2,)
    #   x.amax(dim=0, keepdim=True) -> tensor([[4, 5, 6]]) shape: (1, 3)
    #   x.amax(dim=0, keepdim=False) -> tensor([4, 5, 6]) shape: (3,)

    x = x - x.amax(dim=dim, keepdim=True)

    x = torch.exp(x)
    x = x / x.sum(dim=dim, keepdim=True)
    return x


# glue code for `uv run pytest -k test_softmax_matches_pytorch`
def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """

    return softmax(x=in_features, dim=dim)
