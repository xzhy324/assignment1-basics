import torch
from torch import Tensor
from jaxtyping import Float, Int


class Embedding(torch.nn.Module):
    """A simple embedding layer."""

    weight: Float[torch.nn.Parameter, "num_embeddings embedding_dim"]

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """Construct an embedding module.
        Args:
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors, i.e., 𝑑model
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        # init with 𝒩︀(𝜇 = 0, 𝜎2 = 1) truncated at [−3, 3]
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs.
        Args:
            token_ids: torch.Tensor Input tensor of shape (batch_size, sequence_length)
        Returns:
            torch.Tensor Output tensor of shape (batch_size, sequence_length, embedding_dim)
        """
        # 提取token_ids对应的embedding向量
        # 这里需要这样理解pytorch tensor的 a[b]:相当于b.map(a)，即对b中最细粒度的每个元素（标量），将a[元素]这个tensor，替换掉原本的元素
        # 所以本质上来说，就是对token_ids （batch_size, sequence_length）中每个元素（标量）进行embedding lookup，
        # 得到一个新的tensor，维度为(batch_size, sequence_length, embedding_dim)
        return self.weight[token_ids]


# glue code for `uv run pytest -k test_embedding`
def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    embedding_layer = Embedding(vocab_size, d_model)
    embedding_layer.load_state_dict({"weight": weights})
    return embedding_layer.forward(token_ids)