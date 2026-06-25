import torch
from torch import Tensor
from jaxtyping import Float, Int, Bool
import einops

from cs336_basics.modules.rope import RotaryPositionalEmbedding
from cs336_basics.functions.scaled_dot_product_attention import (
    scaled_dot_product_attention as attention,
)


class MultiheadSelfAttention(torch.nn.Module):
    """A multi-head self-attention layer."""

    # W[:d_model]:Q
    # W[d_model,d_model*2]:K
    # W[d_model*2:]:V
    d_model: int
    num_heads: int
    weight_qkv: Float[Tensor, "d_model_x3 d_model"]
    weight_o: Float[Tensor, "d_model_out d_model_in"]
    rope_module: RotaryPositionalEmbedding | None

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope_module: RotaryPositionalEmbedding | None = None,
        device=None,
        dtype=None,
    ):
        """Construct a multi-head self-attention module.
        Following Vaswani et al., 2017 , set h = d_k = d_v = d_model/num_heads .
        Args:
            d_model: int final dimension of the input
            num_heads: int number of attention heads
            rope_module: Optional[RotaryPositionalEmbedding] module to apply RoPE to Q and K
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        assert d_model % num_heads == 0, "num_heads must divide d_model"
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope_module = rope_module
        self.weight_qkv = torch.nn.Parameter(
            torch.empty((3 * d_model, d_model), device=device, dtype=dtype)
        )
        self.weight_o = torch.nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )

    def forward(
        self,
        x: Tensor,
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ):
        """
        Args:
            x: torch.Tensor Input tensor of shape (... sequence_length, d_model)
            token_positions: Optional tensor with the positions of the tokens, shape (... sequence_length)
        Returns:
            torch.Tensor Output tensor of shape (... sequence_length, d_model)
        """
        assert x.shape[-1] == self.d_model
        # 一次聚合的矩阵乘法
        qkv = einops.einsum(
            x,
            self.weight_qkv,
            "... seq d_model, d_modelx3 d_model -> ... seq d_modelx3",
        )
        # 切开结果
        Q, K, V = qkv.split(self.d_model, dim=-1)
        # rearrange成多头
        Q = einops.rearrange(
            Q,
            "... seq (num_heads d_head) -> ... num_heads seq d_head",
            num_heads=self.num_heads,
        )
        K = einops.rearrange(
            K,
            "... seq (num_heads d_head) -> ... num_heads seq d_head",
            num_heads=self.num_heads,
        )
        V = einops.rearrange(
            V,
            "... seq (num_heads d_head) -> ... num_heads seq d_head",
            num_heads=self.num_heads,
        )

        if self.rope_module:
            assert (
                token_positions is not None
            ), "token_positions must be provided when using RoPE"
            Q = self.rope_module(Q, token_positions=token_positions)
            K = self.rope_module(K, token_positions=token_positions)

        seq_len = x.shape[-2]
        mask = MultiheadSelfAttention._get_mask(seq_len, device=x.device)
        # 计算masked attention （这个mask矩阵代表所有位置都只attend到自己之前的）
        result = attention(Q, K, V, mask)
        # 合并多头
        result = einops.rearrange(
            result, "... num_heads seq d_head -> ... seq (num_heads d_head)"
        )
        # 这里的Wo^{d_model, d_model}的作用是
        # 将机械拼接的多头结果重新通过Wo学习一个混合的方式
        return einops.einsum(
            result,
            self.weight_o,
            "... d_model_in, d_model_out d_model_in -> ... d_model_out",
        )

    # FIXME:这张表也许可以持久化？不需要在每次forward的时候都计算一遍，但可能不是主要瓶颈
    @classmethod
    def _get_mask(cls, seq_len: int, device=None) -> Bool[Tensor, "seq_len seq_len"]:
        """return a (seq_len,seq_len) mask matrix,
        seq_len = 4,looks like:
        [
            [T, F, F, F],
            [T, T, F, F],
            [T, T, T, F],
            [T, T, T, T],
        ]
        Args:
            seq_len: int length of the sequence
            device: torch.device | None = None Device to store the mask on
        Returns:
            torch.BoolTensor Mask tensor of shape (seq_len, seq_len)
        """
        idx = torch.arange(seq_len, device=device)
        row = einops.rearrange(idx, "n -> n 1")
        col = einops.rearrange(idx, "n -> 1 n")
        return col <= row


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection (out_features, in_features)
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection (out_features, in_features)
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection (out_features, in_features)
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    mha = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads)
    weight_qkv = torch.cat(
        [q_proj_weight, k_proj_weight, v_proj_weight],
        dim=0,
    )  # concat the output dimension(dim=0)

    mha.load_state_dict({"weight_qkv": weight_qkv, "weight_o": o_proj_weight})
    return mha(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    rope = RotaryPositionalEmbedding(
        theta=theta, d_k=d_model // num_heads, max_seq_len=max_seq_len
    )
    mha = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, rope_module=rope)
    weight_qkv = torch.cat(
        [q_proj_weight, k_proj_weight, v_proj_weight],
        dim=0,
    )  # concat the output dimension(dim=0)

    mha.load_state_dict({"weight_qkv": weight_qkv, "weight_o": o_proj_weight})
    return mha(in_features, token_positions=token_positions)
