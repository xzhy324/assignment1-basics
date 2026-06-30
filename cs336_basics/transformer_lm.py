import torch
from torch import Tensor
from jaxtyping import Float, Int
from typing import cast

from cs336_basics.modules.embedding import Embedding
from cs336_basics.modules.rope import RotaryPositionalEmbedding
from cs336_basics.modules.transformer_block import TransformerBlock
from cs336_basics.modules.rmsnorm import RMSNorm
from cs336_basics.modules.linear import Linear
from cs336_basics.functions.softmax import softmax


class TransformerLM(torch.nn.Module):
    """A Transformer language model that uses RoPE for positional embeddings."""

    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    vocab_size: int
    context_length: int
    input_embedding: Embedding
    rope_module: RotaryPositionalEmbedding
    transformer_blocks: torch.nn.ModuleList # TransformerBlock
    final_layer_norm: RMSNorm
    lm_head: Linear

    device: torch.device | None
    dtype: torch.dtype | None

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        """Construct a Transformer language model.
        Args:
            vocab_size: int Size of the vocabulary
            context_length: int Maximum number of tokens to process at once
            d_model: int Dimensionality of the model embeddings and sublayer outputs
            num_layers: int Number of Transformer layers to use
            num_heads: int Number of heads to use in multi-headed attention. `d_model`
                must be evenly divisible by `num_heads`.
            d_ff: int Dimensionality of the feed-forward inner layer
            rope_theta: float The RoPE $\\Theta$ parameter
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.device = device
        self.dtype = dtype
        self.input_embedding = Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype
        )
        self.rope_module = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_k=d_model // num_heads,
            max_seq_len=context_length,
            device=device,
        )
        self.transformer_blocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(
                TransformerBlock(
                    d_model, num_heads, d_ff, self.rope_module, device=device, dtype=dtype
                )
            )
        self.final_layer_norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(
            in_features=d_model, out_features=vocab_size, device=device, dtype=dtype
        )

    def forward(self, x: Tensor):
        """Run a forward pass of the Transformer language model.
        Args:
            x: torch.Tensor Input tensor of shape (batch_size, sequence_length)
        Returns:
            torch.Tensor Output tensor of shape (batch_size, sequence_length, vocab_size)
        """
        x = self.input_embedding.forward(x)  # (B,T) -> (B,T,D)
        for transformer_block in self.transformer_blocks:
            block: TransformerBlock = cast(TransformerBlock, transformer_block)
            x = block.forward(x)  # (B,T,D) -> (B,T,D)
        x = self.final_layer_norm.forward(x)
        x = self.lm_head.forward(x)  # (B,T,D) -> (B,T,V)
        return x

    def load_weights(self, weights: dict[str, Tensor]):
        """Load the weights of the Transformer language model from a state dict.
        Args:
            weights: dict[str, Tensor] State dict of our reference implementation. {num_layers}
                refers to an integer between `0` and `num_layers - 1` (the layer index).
        """
        self.input_embedding.load_state_dict(
            {"weight": weights["token_embeddings.weight"]}
        )
        for i in range(self.num_layers):
            transformer_block_weights = {
                "attn.q_proj.weight": weights[f"layers.{i}.attn.q_proj.weight"],
                "attn.k_proj.weight": weights[f"layers.{i}.attn.k_proj.weight"],
                "attn.v_proj.weight": weights[f"layers.{i}.attn.v_proj.weight"],
                "attn.output_proj.weight": weights[
                    f"layers.{i}.attn.output_proj.weight"
                ],
                "ln1.weight": weights[f"layers.{i}.ln1.weight"],
                "ffn.w1.weight": weights[f"layers.{i}.ffn.w1.weight"],
                "ffn.w2.weight": weights[f"layers.{i}.ffn.w2.weight"],
                "ffn.w3.weight": weights[f"layers.{i}.ffn.w3.weight"],
                "ln2.weight": weights[f"layers.{i}.ln2.weight"],
            }
            block: TransformerBlock = cast(TransformerBlock, self.transformer_blocks[i])
            block.load_weights(transformer_block_weights)
        self.final_layer_norm.load_state_dict({"weight": weights["ln_final.weight"]})
        self.lm_head.load_state_dict({"weight": weights["lm_head.weight"]})


# glue code for `uv run pytest -k test_transformer_lm`
def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """

    transformer_lm = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )
    transformer_lm.load_weights(weights)
    return transformer_lm.forward(in_indices)
