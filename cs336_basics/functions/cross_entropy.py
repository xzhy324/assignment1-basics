import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    inputs = inputs - inputs.amax(dim=-1, keepdim=True)  # avoid exp overflow
    index = targets[:, None]  # (B,) -> (B,1)
    # gather的output和index形状相同
    correct_value = inputs.gather(
        dim=-1, index=index
    )  # (B,V).index(dim=-1, (B,1)) => (B,1)
    correct_value = correct_value.squeeze(-1)  # (B,1) => (B)

    x = torch.exp(inputs).sum(dim=-1)  # ∑exp(x) : (B,V) => (B)
    x = torch.log(x) - correct_value  # ln∑exp(x) - real_answer
    return x.mean(dim=0)  # (B,) => (,)


# glue code for `uv run pytest -k test_cross_entropy`
def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    return cross_entropy(inputs, targets)
