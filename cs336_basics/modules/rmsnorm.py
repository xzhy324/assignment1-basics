import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import einops


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm normalizes the input over the last dimension without subtracting the mean:

    Given a vector 𝑎 ∈ ℝ^𝑑model of activations, RMSNorm will rescale each activation 𝑎𝑖 as follows

        RMSNorm(a_i) = (a_i / RMS(a)) * g_i

    where:

        RMS(a) = sqrt((1 / d_model) * sum_{i=1}^{d_model}(a_i^2) + eps)

    # g_i is a learnable gain parameter, with one parameter for each hidden dimension, so g has shape (d_model,).
    # eps is a small constant for numerical stability, usually fixed to 1e-5.
    """

    d_model: int
    eps: float = 1e-5
    weight: Float[Tensor, "d_model"]

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        """
        Construct the RMSNorm module.

        Args:
            d_model: Hidden dimension of the model.
            eps: Epsilon value for numerical stability.
            device: Device to store the parameters on.
            dtype: Data type of the parameters.
        """
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.weight = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input tensor.

        Args:
            x: Input tensor of shape (... d_model).

        Returns:
            Tensor of the same shape as x.
        """
        assert x.shape[-1] == self.d_model, f"Expected last dimension of input to be {self.d_model}, but got {x.shape[-1]}"
        original_dtype = x.dtype
        x = x.to(torch.float32)
        # 这里1表示reduce之后创建一个长度为1的匿名维度，这相当于最后一维变成一个常量
        # (b,t,d)的原始tensor和(b,t,1)的tensor运算时，1维度上的内容会广播给d维度
        square_mean = einops.reduce(x**2, "... d_model -> ... 1", "mean")
        rms = torch.sqrt(square_mean + self.eps)
        # 一般的，elementwise 乘法成立的条件是：两个 tensor 的 shape 从右往左逐维比较，每一维必须满足下面三种情况之一：
        # 维度相等
        # 其中一个维度是 1
        # 其中一个 tensor 在这一维不存在

        # 所以这些都可以：

        # (batch, seq, d_model) * (d_model,)
        # (batch, seq, d_model) * (1, d_model)
        # (batch, seq, d_model) * (batch, seq, 1)
        # (batch, seq, d_model) * (batch, seq, d_model)
        result = x / rms * self.weight
        return result.to(original_dtype)


# glue code for 'uv run pytest -k test_rmsnorm'
def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """

    rmsnorm_layer = RMSNorm(d_model=d_model, eps=eps)
    rmsnorm_layer.load_state_dict({"weight": weights})
    return rmsnorm_layer.forward(in_features)
