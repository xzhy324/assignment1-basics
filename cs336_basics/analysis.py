import os
import resource
import time


def log_mem(label: str):
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    print(
        f"[{timestamp} mem pid={os.getpid()}] {label}: maxrss={rss_kb / 1024:.1f} MiB",
        flush=True,
    )


def count_trainable_parameters(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    d_ff: int,
):
    def count_tranformer_block_params(d_model, d_ff):
        # rmsnorm + attention + rmsnorm + ffn
        rmsnorm = d_model
        attention = (
            d_model * 3 * d_model + d_model * d_model
        )  # weight_qkv + weight_output
        ffn = d_model * d_ff + d_model * d_ff + d_model * d_ff  # 2个升维，1个降维
        return rmsnorm * 2 + attention + ffn

    embedding = vocab_size * d_model
    tranformer_blocks = num_layers * count_tranformer_block_params(d_model, d_ff)
    final_layer_norm = d_model
    lm_head = vocab_size * d_model

    return embedding + tranformer_blocks + final_layer_norm + lm_head


def count_matmul_flops(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
):
    """Count the FLOPs of matrix multiplications for a Transformer LM.
    Assume that the model is run in inference mode, so no backward pass is counted.
    and no batch dimension is counted. The input is a single sequence of length `context_length`.
    """
    D = d_model
    T = context_length
    V = vocab_size

    def matmul_flops(m, n, p):
        """Count the FLOPs of a matrix multiplication of shape (m,n) @ (n,p)."""
        return 2 * m * n * p

    def count_transformer_block_matmul_flops(D, num_heads, d_ff, T):
        """Count the FLOPs of matrix multiplications for a single Transformer block.
        Args:
            D: int, the model dimension
            num_heads: int, the number of attention heads
            d_ff: int, the dimension of the feedforward layer
            T: int, the sequence length
        Returns:
            int, the total FLOPs of matrix multiplications for a single Transformer block
        """
        attention = (
            matmul_flops(T, D, 3 * D)  # x @ W_qkv
            + num_heads * matmul_flops(T, D // num_heads, T)  # Q @ K^T
            + num_heads * matmul_flops(T, T, D // num_heads)  # attention @ V
            + matmul_flops(T, D, D)  # attention_output @ W_o
        )
        ffn = (
            matmul_flops(T, D, d_ff) * 2   # x @ W1 and x @ W3
            + matmul_flops(T, d_ff, D)  # ffn_output @ W2
        )
        return attention + ffn

    transformer_blocks = num_layers * count_transformer_block_matmul_flops(D, num_heads, d_ff, T)
    final_layer_norm = 0  # RMSNorm is elementwise, so arithmetic intensity is not increased as the matrix size increases
    lm_head = matmul_flops(T, D, V)  # lm_head @ W
    return transformer_blocks + final_layer_norm + lm_head


if __name__ == "__main__":
    # for parameter counting
    vocab_size = 50257
    d_model = 1600
    num_layers = 48
    d_ff = 4288
    # additionally, for FLOPs counting, we need to specify the context length and number of attention heads
    context_length = 1024
    num_heads = 25

    print(
        "Counting trainable parameters for a Transformer LM with the following configuration:"
    )
    print(f"Vocab size: {vocab_size}")
    print(f"Model dimension (d_model): {d_model}")
    print(f"Number of layers: {num_layers}")
    print(f"Feedforward dimension (d_ff): {d_ff}")

    total_params = count_trainable_parameters(vocab_size, d_model, num_layers, d_ff)
    print(f"\nTotal trainable parameters: {total_params:,}")
    print(f"Total trainable parameters (in millions): {total_params / 1_000_000:.2f}M")


    total_memory_bytes = total_params * 4  # float32 is 4 bytes
    total_memory_gb = total_memory_bytes / (1024**3)
    print(
        f"\nif all the parameters are stored as fp32, the total memory would be:{total_memory_gb:.2f} GB"
    )

    flops = count_matmul_flops(
        vocab_size, context_length, d_model, num_layers, num_heads, d_ff
    )
    print(f"\nTotal FLOPs: {flops:,}")
    print(f"Total FLOPs (in billions): {flops / 1_000_000_000:.2f}B")
