import torch
import math
from jaxtyping import Bool, Float, Int
from torch import Tensor
import einops

#   当你要构造一个矩阵/张量，且元素形如：

#   A[i, k] = 某个关于 i 和 k 的公式

#   不要先想“双重 for 怎么写”，先想：

#   1. 输出有哪些轴？
#   2. 每个轴的下标代表什么？
#   3. 标量公式里，下标之间是可分离依赖还是不可分离依赖？
#   4. 能不能用 broadcasting 表达坐标组合？
#   5. 是否真的需要显式 materialize 整张坐标表？

#   可分离依赖

#   如果公式能写成：
#   A[i, k] = h(f(i), g(k))
#   其中 f(i) 只依赖行坐标，g(k) 只依赖列坐标，那就优先用 broadcasting。

#   思维模式：
#   - 构造行坐标向量：形状从 (N,) 变成 (N, 1)
#   - 构造列坐标向量：形状从 (M,) 变成 (1, M)
#   - 让 PyTorch 自动广播成 (N, M)
#   - 对广播结果做 elementwise 运算

#   RoPE angle 就是典型例子：
#   angle[i, k] = position[i] * inv_freq[k]
#   虽然它同时依赖 i 和 k，但依赖结构是可分离的，所以不需要 meshgrid。

#   不可分离依赖

#   如果公式里出现真正的坐标关系，例如：
#   i - k
#   abs(i - k)
#   i + k
#   distance(i, k)
#   same_bucket(i, k)
#   within_window(i, k)
#   那就是不可分离或不方便分离的依赖。这时用 coordinate grid / meshgrid 思维。

#   思维模式：
#   - 构造一张 I 表，每个位置存自己的行坐标
#   - 构造一张 K 表，每个位置存自己的列坐标
#   - 然后像写标量公式一样写整张表的公式

#   比如 attention 里的相对距离：
#   bias[q, k] = -abs(q - k)
#   这里不能只靠单独的 f(q) 和 g(k) 表达，需要看两个坐标之间的关系，所以用坐标网格思考。


#   一句话笔记
#   先把矩阵元素看成坐标函数 A[i,k]；能拆成各轴独立函数就用 broadcasting，涉及相对位置/距离/比较就用 coordinate grid。不要从循环填表出发，要从“坐标轴如何组合成整张表”出发

class RotaryPositionalEmbedding(torch.nn.Module):
    """
    Rotary Positional Embedding (RoPE).

    This module applies RoPE to an input tensor of shape:

        (..., seq_len, d_k)

    where:
        ...     : arbitrary batch dimensions
        seq_len : sequence length
        d_k     : dimension of each query/key vector

    token_positions has shape:

        (..., seq_len)

    and specifies the absolute token positions corresponding to x.

    For a given query token 𝑞(𝑖) = 𝑊_𝑞·𝑥(𝑖) ∈ ℝ^𝑑 at token position 𝑖,
    we will apply a pairwise rotation matrix 𝑅𝑖, giving us 𝑞′(𝑖) = 𝑅𝑖·𝑞(𝑖) = 𝑅𝑖·𝑊_𝑞·𝑥(𝑖).
    Here, 𝑅𝑖 will rotate pairs of embedding elements 𝑞(𝑖)_{2𝑘−1:2𝑘} as 2d vectors
    by the angle
        𝜃{𝑖,𝑘} = i / Θ^[(2𝑘−2)/𝑑]
    for 𝑘 ∈ {1, …, 𝑑/2} and some constant Θ.

    Thus, we can consider 𝑅𝑖 to be a block-diagonal matrix of size 𝑑 × 𝑑,
    with blocks 𝑅_{𝑖,𝑘} for 𝑘 ∈{1, …, 𝑑/2},
    with 𝑅_{𝑖,𝑘} = [
         cos(𝜃{𝑖,𝑘})  sin(𝜃{𝑖,𝑘})
        −sin(𝜃{𝑖,𝑘})  cos(𝜃{𝑖,𝑘})
    ]
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct the RoPE module.

        Args:
            theta: Θ value for RoPE.
            d_k: Dimension of query/key vectors.
            max_seq_len: Maximum sequence length.
            device: Device to store precomputed buffers on.
            dtype: Data type for the precomputed buffers.
        """
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        assert d_k % 2 == 0, f"input d_k:{d_k} is not even!"

        cos = torch.empty((max_seq_len, d_k // 2), device=device, dtype=dtype)
        sin = torch.empty((max_seq_len, d_k // 2), device=device, dtype=dtype)

        # 𝜃{𝑖,𝑘} = i / Θ^{2𝑘/𝑑}
        # first dimension初始化为0到i-1
        seq_dim = torch.arange(0, max_seq_len, 1, device=device, dtype=dtype)
        d_k_dim = torch.arange(0, d_k // 2, 1, device=device, dtype=dtype)
        # second dim初始化为 Θ^{2𝑘/𝑑}
        d_k_dim = self.theta ** (d_k_dim / (d_k // 2))
        # reshape以应用广播规则
        seq_dim = einops.rearrange(seq_dim, "m -> m 1")
        d_k_dim = einops.rearrange(d_k_dim, "n -> 1 n")
        # broadcast运算
        # shape: (m,1) / (1,n) => (m,n)
        angle = seq_dim / d_k_dim

        cos = torch.cos(angle)
        sin = torch.sin(angle)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply RoPE to x.

        Args:
            x: Tensor of shape (..., seq_len, d_k).
            token_positions: Tensor of shape (..., seq_len).

        Returns:
            Tensor of the same shape as x.
        """
        # 这里token positions的seq_len维度形如（0，1，2，3，..., seq_len-1），就是一个连续的index
        # 这个量不能默认从0开始。api应设计为能由用户指定:
        # 例如自回归生成时，前面已经有 5 个 token，现在只输入新生成的 3 个 token：
        # x.shape == (batch, 3, d_k)
        # 但它们在完整上下文里的位置应该是：
        # token_positions = torch.tensor([
        #     [5, 6, 7]
        # ])
        # 而不是：
        # [0, 1, 2]
        # 这对 KV cache 特别重要。否则模型会把新 token 当成序列开头来旋转，位置信息就错了。
        assert x.shape[-1] == self.d_k  # 保证了x最后一维长度为偶数

        # shape: (..., d_k/2)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # shape:将token_positons(...,seq_len)按cos表(max_seq_len,d_k/2)索引为
        # (...,seq_len,d_k/2)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # x_odd.shape= (...,seq_len,d_k/2)与cos.shape=(...,seq_len,d_k/2) 逐元素乘
        result_even = x_even * cos - x_odd * sin
        result_odd = x_even * sin + x_odd * cos

        # shape (2,...,d_k/2)
        stacked = [result_even, result_odd]
        # shape (...,d_k)
        return einops.rearrange(stacked, "two ... d_half_k -> ... (d_half_k two)")


# glue code for `uv run pytest -k test_rope`
def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len)
    return rope.forward(x=in_query_or_key, token_positions=token_positions)
