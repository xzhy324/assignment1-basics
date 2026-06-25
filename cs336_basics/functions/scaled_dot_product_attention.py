import math

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import einops

from cs336_basics.functions.softmax import softmax


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
            if True, the corresponding query-key IS ATTENDED, if False, the corresponding query-key is NOT ATTENDED
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.shape[-1]
    dot_product = einops.einsum(
        Q, K, "... queries d_k, ... keys d_k -> ... queries keys"
    )
    dot_product = dot_product / math.sqrt(d_k)
    if mask is not None:
        mask = ~mask
        dot_product = dot_product.masked_fill(mask=mask, value=float("-inf"))
    dot_product = softmax(dot_product, dim=-1)
    return einops.einsum(
        dot_product, V, "... queries keys, ... keys d_v -> ... queries d_v"
    )


# glue code for
#    uv run pytest -k test_scaled_dot_product_attention
# and
#    uv run pytest -k test_4d_scaled_dot_product_attention
def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    return scaled_dot_product_attention(Q, K, V, mask)
