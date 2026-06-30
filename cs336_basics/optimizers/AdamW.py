from collections.abc import Callable
from typing import Optional, cast
from math import sqrt

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        try:
            beta1, beta2 = betas
        except ValueError as e:
            raise ValueError("betas must be a pair") from e
        if not 0 <= beta1 < 1:
            raise ValueError(f"Invalid beta parameter at index 0: {beta1}")
        if not 0 <= beta2 < 1:
            raise ValueError(f"Invalid beta parameter at index 1: {beta2}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params=params, defaults=defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            b1, b2 = group["betas"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                p = cast(torch.Tensor, p)
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 1)  # Get iteration number from the state, or 1.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                lr_t = (
                    lr * sqrt(1 - b2**t) / (1 - b1**t)
                )  # Compute adjusted 𝛼 for iteration 𝑡
                p.data -= lr * weight_decay * p.data  # Apply weight decay
                m = state.get(
                    "first_moment",
                    torch.zeros(
                        p.shape, device=p.device, dtype=p.dtype
                    ),  # TODO: 优化器参数的dtype可以不与param一致
                )
                v = state.get(
                    "second_moment",
                    torch.zeros(
                        p.shape, device=p.device, dtype=p.dtype
                    ),  # TODO: 优化器参数的dtype可以不与param一致
                )
                m = b1 * m + (1 - b1) * grad  # Update the first moment estimate
                v = b2 * v + (1 - b2) * (grad**2)  # Update the second moment estimate
                p.data -= (
                    lr_t * m / (torch.sqrt(v) + eps)
                )  # Apply moment-adjusted weight updates
                state["t"] = t + 1  # Increment iteration number.
                state["first_moment"] = m
                state["second_moment"] = v
        return loss
