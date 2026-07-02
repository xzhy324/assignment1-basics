from collections.abc import Iterable
import torch


# $ uv run pytest -k test_gradient_clipping
def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6
) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    parameters = list(
        parameters
    )  # in case the input is a generator, we need to convert it to a list to iterate multiple times
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm**0.5

    clip_coef = max_l2_norm / (total_norm + eps)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                # let pytorch know that we don't want to track this operation in the autograd graph
                # but we want to modify the gradients in-place
                # 1. DO NOT directly assign p.grad = p.grad * clip_coef, because that would create a new tensor and break the computation graph
                # 2. DO NOT use p.grad.data.mul_(clip_coef), because that would modify the underlying data of the tensor, which is not recommended
                with torch.no_grad():
                    p.grad.mul_(clip_coef)
