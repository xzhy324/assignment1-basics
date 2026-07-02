import torch
from torch import Tensor
from jaxtyping import Float, Int, Bool
import einops


from cs336_basics.modules.rope import RotaryPositionalEmbedding
from cs336_basics.modules.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.modules.rmsnorm import RMSNorm
from cs336_basics.modules.swiglu import SwiGLU


class TransformerBlock(torch.nn.Module):
    """A pre-norm Transformer block."""

    d_model: int
    num_heads: int
    d_ff: int

    attention: MultiheadSelfAttention
    ffn: SwiGLU
    first_rmsnorm: RMSNorm
    second_rmsnorm: RMSNorm

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_module: RotaryPositionalEmbedding,
        device=None,
        dtype=None,
    ):
        """Construct a pre-norm Transformer block.
        Args:
            d_model: int final dimension of the input
            num_heads: int number of attention heads
            d_ff: int dimension of the feed-forward inner layer
            rope_module: RotaryPositionalEmbedding module to apply RoPE to Q and K
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        assert d_model % num_heads == 0, "num_heads must divide d_model"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attention = MultiheadSelfAttention(
            d_model, num_heads, rope_module, device=device, dtype=dtype
        )
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.first_rmsnorm = RMSNorm(d_model, device=device, dtype=dtype)
        self.second_rmsnorm = RMSNorm(d_model, device=device, dtype=dtype)

    def load_weights(self, weights: dict[str, Tensor]):
        """load the weights of a pre-norm Transformer block from a state dict of our reference implementation.
        Args:
            weights (dict[str, Tensor]):
                State dict of our reference implementation.
                The keys of this dictionary are:
                - `attn.q_proj.weight`
                    The query projections for all `num_heads` attention heads.
                    Shape is (d_model, d_model).
                    The rows are ordered by matrices of shape (num_heads, d_k),
                    so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
                - `attn.k_proj.weight`
                    The key projections for all `num_heads` attention heads.
                    Shape is (d_model, d_model).
                    The rows are ordered by matrices of shape (num_heads, d_k),
                    so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
                - `attn.v_proj.weight`
                    The value projections for all `num_heads` attention heads.
                    Shape is (d_model, d_model).
                    The rows are ordered by matrices of shape (num_heads, d_v),
                    so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
                - `attn.output_proj.weight`
                    Weight of the multi-head self-attention output projection
                    Shape is (d_model, d_model).
                - `ln1.weight`
                    Weights of affine transform for the first RMSNorm
                    applied in the transformer block.
                    Shape is (d_model,).
                - `ffn.w1.weight`
                    Weight of the first linear transformation in the FFN.
                    Shape is (d_ff, d_model).
                - `ffn.w2.weight`
                    Weight of the second linear transformation in the FFN.
                    Shape is (d_model, d_ff).
                - `ffn.w3.weight`
                    Weight of the third linear transformation in the FFN.
                    Shape is (d_ff, d_model).
                - `ln2.weight`
                    Weights of affine transform for the second RMSNorm
                    applied in the transformer block.
                    Shape is (d_model,).
        """
        weight_qkv = torch.cat(
            [
                weights["attn.q_proj.weight"],
                weights["attn.k_proj.weight"],
                weights["attn.v_proj.weight"],
            ],
            dim=0,
        )
        self.attention.load_state_dict(
            {"weight_qkv": weight_qkv, "weight_o": weights["attn.output_proj.weight"]}
        )

        self.first_rmsnorm.load_state_dict({"weight": weights["ln1.weight"]})
        self.second_rmsnorm.load_state_dict({"weight": weights["ln2.weight"]})
        self.ffn.load_state_dict(
            {
                "w1": weights["ffn.w1.weight"],
                "w2": weights["ffn.w2.weight"],
                "w3": weights["ffn.w3.weight"],
            }
        )

    def forward(
        self,
        x: Tensor,
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Tensor:
        """compute the output of a pre-norm Transformer block.
        Args:
            x: Tensor has shape of (... d_model)
            token_positions: Tensor has shape of (... sequence_length) | None
                if None, then the token positions are initialized to be (sequence_length,)
        Returns:
            Tensor has shape of (... d_model)
        """
        if token_positions is not None:
            assert x.shape[-2] == token_positions.shape[-1], "sequence_length mismatch"
        else:
            token_positions = torch.arange(
                x.shape[-2], device=x.device, dtype=torch.int64
            ) # shape: (sequence_length)
        x = x + self.attention(self.first_rmsnorm(x), token_positions=token_positions)
        x = x + self.ffn(self.second_rmsnorm(x))
        return x


# glue code for `uv run pytest -k test_transformer_block`
def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    rope_module = RotaryPositionalEmbedding(
        theta=theta,
        d_k=d_model // num_heads,
        max_seq_len=max_seq_len,
    )

    tranformer_block = TransformerBlock(
        d_model=d_model, num_heads=num_heads, d_ff=d_ff, rope_module=rope_module
    )

    tranformer_block.load_weights(weights)

    return tranformer_block(in_features)
