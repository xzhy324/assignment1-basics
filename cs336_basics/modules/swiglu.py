import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from einops import einsum


class SwiGLU(torch.nn.Module):
    """
    FFN(𝑥) = SwiGLU(𝑥, 𝑊1, 𝑊2, 𝑊3) = 𝑊2(SiLU(𝑊1𝑥) ⊙ 𝑊3𝑥)
    where
        𝑥 ∈ ℝ𝑑model , 𝑊1, 𝑊3 ∈ ℝ𝑑ff×𝑑model , 𝑊2 ∈ ℝ𝑑model×𝑑ff , and canonically, 𝑑ff = 8/3𝑑model.
    """

    d_model: int
    d_ff: int
    w1: Float[Tensor, " d_ff d_model"]
    w2: Float[Tensor, " d_model d_ff"]
    w3: Float[Tensor, " d_ff d_model"]

    def __init__(
        self,
        d_model: int,
        d_ff: int | None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        if d_ff == None:
            raw = int(d_model * 8 / 3)
            self.d_ff = ((raw + 63) // 64) * 64  # find the nearest that 64 devides
        else:
            self.d_ff = d_ff

        self.w1 = torch.nn.Parameter(
            torch.empty((self.d_ff, self.d_model), device=device, dtype=dtype)
        )
        self.w3 = torch.nn.Parameter(
            torch.empty((self.d_ff, self.d_model), device=device, dtype=dtype)
        )
        self.w2 = torch.nn.Parameter(
            torch.empty((self.d_model, self.d_ff), device=device, dtype=dtype)
        )

        # initialize the linear weights using truncated normal distribution
        # 𝒩︀(𝜇 = 0, 𝜎2 = 2/(𝑑in+𝑑out) ) truncated at [−3𝜎, 3𝜎].
        std = (2 / (self.d_ff + self.d_model)) ** 0.5
        torch.nn.init.trunc_normal_(self.w1, mean=0.0, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.w3, mean=0.0, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.w2, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor has shape of (... d_model)
        Returns:
            Tensor: same shape of x
        """

        xw1 = einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff")
        xw3 = einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")
        return einsum(
            _silu(xw1) * xw3, self.w2, "... d_ff, d_model d_ff -> ... d_model"
        )


def _silu(x: Tensor):
    # return x / (1 + torch.exp(-x))
    return torch.nn.SiLU().forward(x)


# glue code for `uv run pytest -k test_swiglu`
def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    ffn_layer = SwiGLU(d_model=d_model, d_ff=d_ff)
    ffn_layer.w1.data = w1_weight
    ffn_layer.w2.data = w2_weight
    ffn_layer.w3.data = w3_weight
    return ffn_layer.forward(in_features)
