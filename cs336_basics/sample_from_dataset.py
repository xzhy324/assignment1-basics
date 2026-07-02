import numpy.typing as npt
import numpy as np
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # 先随机采样 batch_size 个 start index，再对每个 start index 构造长度为 context_length 的连续窗口，
    # 最后把 sampled 和右移一位的 sampled_plus_one 这两个cpu array 变成指定device的tensor。
    start_indexes = np.random.randint(0, len(dataset) - context_length, size=batch_size)

    sampled = np.stack([dataset[i : i + context_length] for i in start_indexes])
    sampled_plus_one = np.stack(
        [dataset[i + 1 : i + context_length + 1] for i in start_indexes]
    )

    sampled_tensor = torch.tensor(sampled, dtype=torch.long, device=device)
    sampled_plus_one_tensor = torch.tensor(
        sampled_plus_one, dtype=torch.long, device=device
    )

    return sampled_tensor, sampled_plus_one_tensor
