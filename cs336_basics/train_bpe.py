import os
from concurrent.futures import ProcessPoolExecutor
import regex as re
from cs336_basics.pretokenization import find_chunk_boundaries


def _count_word_like_pieces(
    args: tuple[str, int, int, list[bytes]],
) -> dict[tuple[bytes, ...], int]:
    """Given a chunk of the input file defined by start and end byte offsets,
    return a dictionary mapping each word-like piece (represented as a tuple of byte tokens) in that chunk to its frequency count.
    The special tokens should be treated as separate pieces and should not be split.
    Args:
        args (tuple[str, int, int, list[bytes]]): A tuple containing:
            - input_path (str): The path to the input file.
            - start (int): The starting byte offset of the chunk.
            - end (int): The ending byte offset of the chunk.
            - special_tokens_bytes (list[bytes]): A list of special tokens represented as bytes. These should be treated as separate pieces and not split.
    Returns:
        dict[tuple[bytes, ...], int]: A dictionary mapping each word-like piece (represented as a tuple of byte tokens) to its frequency count in the chunk.
    """
    input_path, start, end, special_tokens_bytes = args
    chunk_word_like_pieces: dict[tuple[bytes, ...], int] = {}
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # 先按照special tokens将大chunk直接切分成多个小chunk
    special_tokens_pattern = b"|".join(
        re.escape(token) for token in special_tokens_bytes
    ).decode("utf-8")
    smaller_chunks = re.split(special_tokens_pattern, chunk)

    # 对每个小chunk，按照PATTERN切分成word-like pieces，并统计这些piece的频次
    piece_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  # GPT2的pretokenizer使用的pattern
    for smaller_chunk in smaller_chunks:
        for match in re.finditer(piece_pattern, smaller_chunk):
            piece = match.group(0).encode("utf-8")
            byte_tokens = tuple(
                piece[i : i + 1] for i in range(len(piece))
            )  # 将piece切分成一个个byte token，组成一个tuple作为key
            chunk_word_like_pieces[byte_tokens] = (
                chunk_word_like_pieces.get(byte_tokens, 0) + 1
            )
    return chunk_word_like_pieces


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    # initialize vocab with special tokens and 0-255 byte tokens
    for i in range(256):
        vocab[len(vocab)] = bytes([i])
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    # 将输入按照并行线程数切分成多个 chunk，每个 chunk 各自做pretokenization，统计 word-like piece 的频次，然后合并统计结果
    special_tokens_bytes = [token.encode("utf-8") for token in special_tokens]
    f = open(input_path, "rb")
    num_processes = max(1, os.cpu_count() - 1)  # leave 1 CPU free
    assert (
        special_tokens_bytes[0] == b"<|endoftext|>"
    ), "The first special token must be <|endoftext|> for the chunk boundary finding logic to work correctly."
    boundaries = find_chunk_boundaries(
        f, desired_num_chunks=num_processes, split_special_token=special_tokens_bytes[0]
    )
    f.close()

    word_like_pieces: list[dict[tuple[bytes, ...], int]] = []  # one per chunk

    # 并行统计每个 chunk 里的 word-like piece 频次
    chunk_ranges = list(zip(boundaries[:-1], boundaries[1:]))
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for chunk_word_like_pieces in executor.map(
            _count_word_like_pieces,
            (
                (str(input_path), start, end, special_tokens_bytes)
                for start, end in chunk_ranges
            ),
        ):
            word_like_pieces.append(chunk_word_like_pieces)

    # 合并所有chunk的统计结果
    total_word_like_pieces: dict[tuple[bytes, ...], int] = {}  # (piece tuple) -> count
    for chunk_word_like_piece in word_like_pieces:
        for old_piece, count in chunk_word_like_piece.items():
            total_word_like_pieces[old_piece] = (
                total_word_like_pieces.get(old_piece, 0) + count
            )

    all_pairs: dict[tuple[bytes, bytes], int] = {}  # (bytes, bytes) -> count
    for old_piece, count in total_word_like_pieces.items():
        for i in range(len(old_piece) - 1):
            pair = (old_piece[i], old_piece[i + 1])
            all_pairs[pair] = all_pairs.get(pair, 0) + count

    # # 添加最频繁的 pairs 到 vocab 和 merges，直到达到 vocab_size
    # while len(vocab) < vocab_size:
    #     # 从all pairs中取出当前最频繁的 pair
    #     pair, count = max(all_pairs.items(), key=lambda x: (x[1],x[0])) # 先按频次排序，频次相同则按pair的字典序排序
    #     # 更新merges和vocab
    #     merges.append(pair)
    #     new_token = pair[0] + pair[1]
    #     vocab[len(vocab)] = new_token
    #     # 构造一个新的total_word_like_pieces，包含所有替换了当前选中的最大pair的piece
    #     updated_word_like_pieces: dict[tuple[bytes, ...], int] = {}
    #     for piece, piece_count in total_word_like_pieces.items():
    #         new_piece = []
    #         i=0 # 这个循环的目的是将当前的piece中所有当前选中的最大pair（如果有的话）替换成new_token，生成一个新的piece
    #         while i < len(piece):
    #             if i < len(piece) - 1 and (piece[i], piece[i+1]) == pair:
    #                 new_piece.append(new_token)
    #                 i += 2
    #             else:
    #                 new_piece.append(piece[i])
    #                 i += 1
    #         new_piece_tuple = tuple(new_piece)
    #         updated_word_like_pieces[new_piece_tuple] = updated_word_like_pieces.get(new_piece_tuple, 0) + piece_count

    #     # 全量重算all_pairs
    #     all_pairs.clear()
    #     for piece, count in updated_word_like_pieces.items():
    #         for i in range(len(piece) - 1):
    #             pair = (piece[i], piece[i+1])
    #             all_pairs[pair] = all_pairs.get(pair, 0) + count
    #     # 更新total_word_like_pieces
    #     total_word_like_pieces = updated_word_like_pieces

    # 上面这个循环的效率比较低，因为每次merge都要全量重算all_pairs。下面这个循环的效率更高，因为每次merge只需要更新包含了当前选中的最大pair的piece对应的pair频次，而不需要全量重算。
    # 我们需要额外维护一个 pairs 到 piece的索引，以便在每次merge后快速找到受影响的piece并更新对应的pair频次。
    pair2pieces: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = (
        {}
    )  # (bytes, bytes) -> unique pieces that contain this pair
    # 初始化这个映射
    for old_piece in total_word_like_pieces:
        for i in range(len(old_piece) - 1):
            pair = (old_piece[i], old_piece[i + 1])
            if pair not in pair2pieces:
                pair2pieces[pair] = set()
            pair2pieces[pair].add(old_piece)

    # 添加最频繁的 pairs 到 vocab 和 merges，直到达到 vocab_size
    while len(vocab) < vocab_size:
        # 从all pairs中取出当前最频繁的 pair
        pair, count = max(
            all_pairs.items(), key=lambda x: (x[1], x[0])
        )  # 先按频次排序，频次相同则按pair的字典序排序
        # 更新merges和vocab, 每次merge都把当前选中的最大pair添加到merges里，把这个pair对应的new_token添加到vocab里
        merges.append(pair)
        new_token = pair[0] + pair[1]
        vocab[len(vocab)] = new_token

        affected_pieces = list(pair2pieces.get(pair, set()))
        for old_piece in affected_pieces:
            assert (
                old_piece in total_word_like_pieces
            ), f"Old piece {old_piece} not found in total_word_like_pieces"
            # 对每个受影响的旧 piece，做以下更新：
            #   step1:从 all_pairs 扣掉这个旧 piece 的所有 pair 贡献。
            #   step2:把旧 piece 中的 best_pair 做左到右非重叠 merge。
            #   step3:把新 piece 的所有 pair 贡献加回 all_pairs。
            #   step4:更新 pair -> pieces 里旧 piece / 新 piece 的关系。
            #   step5:把旧 piece 从 total_word_like_pieces 里删除，把新 piece 加入 total_word_like_pieces。

            # step1
            piece_count = total_word_like_pieces[old_piece]
            for i in range(len(old_piece) - 1):
                old_pair = (old_piece[i], old_piece[i + 1])
                all_pairs[old_pair] -= piece_count
                if all_pairs[old_pair] == 0:
                    del all_pairs[old_pair]

            # step2
            new_piece = []
            i = 0
            while i < len(old_piece):
                if i < len(old_piece) - 1 and (old_piece[i], old_piece[i + 1]) == pair:
                    new_piece.append(new_token)
                    i += 2
                else:
                    new_piece.append(old_piece[i])
                    i += 1
            new_piece_tuple = tuple(new_piece)

            # step3
            for i in range(len(new_piece) - 1):
                new_pair = (new_piece[i], new_piece[i + 1])
                all_pairs[new_pair] = all_pairs.get(new_pair, 0) + piece_count

            # step4
            # 从 pair2pieces里把旧 piece 从所有包含它的 pair 的映射里删除
            unique_pairs_in_old_piece = set(
                (old_piece[i], old_piece[i + 1]) for i in range(len(old_piece) - 1)
            )
            for old_pair in unique_pairs_in_old_piece:
                pair2pieces[old_pair].remove(old_piece)
                if not pair2pieces[
                    old_pair
                ]:  # 如果这个 pair 已经没有任何 piece 了，就把这个 pair 从 pair2pieces里删除
                    del pair2pieces[old_pair]
            # 把新 piece 添加到 pair2pieces里对应 pair 的映射里
            for i in range(len(new_piece) - 1):
                new_pair = (new_piece[i], new_piece[i + 1])
                if new_pair not in pair2pieces:
                    pair2pieces[new_pair] = set()
                pair2pieces[new_pair].add(new_piece_tuple)

            # step5
            del total_word_like_pieces[old_piece]
            total_word_like_pieces[new_piece_tuple] = (
                total_word_like_pieces.get(new_piece_tuple, 0) + piece_count
            )

    return vocab, merges


import json


def serialize_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_filepath: str,
    merges_filepath: str,
):
    """Serialize the vocab and merges to files. The format of these files is up to you, but they should contain all the information needed to reconstruct the vocab and merges in memory."""
    with open(vocab_filepath, "w", encoding="utf-8") as f:
        vocabs_serializable = {
            str(token_id): token_bytes.hex() for token_id, token_bytes in vocab.items()
        }
        json.dump(vocabs_serializable, f)

    with open(merges_filepath, "w", encoding="utf-8") as f:
        merges_serializable = [
            (token1.hex(), token2.hex()) for token1, token2 in merges
        ]
        json.dump(merges_serializable, f)


def deserialize_vocab_and_merges(
    vocab_filepath: str, merges_filepath: str
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Deserialize the vocab and merges from files. The format of these files should be the same as the output of `serialize_vocab_and_merges`."""
    with open(vocab_filepath, "r", encoding="utf-8") as f:
        vocabs_serializable = json.load(f)
        vocab = {
            int(token_id): bytes.fromhex(token_bytes_hex)
            for token_id, token_bytes_hex in vocabs_serializable.items()
        }

    with open(merges_filepath, "r", encoding="utf-8") as f:
        merges_serializable = json.load(f)
        merges = [
            (bytes.fromhex(token1_hex), bytes.fromhex(token2_hex))
            for token1_hex, token2_hex in merges_serializable
        ]

    return vocab, merges


def is_this_in_vocabs(token_bytes: bytes, vocab: dict[int, bytes]) -> bool:
    """Helper function to check if a given token_bytes is already in the vocab."""
    return any(token_bytes == v for v in vocab.values())


if __name__ == "__main__":
    rvocab, rmerges = train_bpe(
        input_path="/Users/daniel/Documents/cs336/homework/assignment1-basics/data.nosync/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )

    serialize_vocab_and_merges(
        rvocab,
        rmerges,
        vocab_filepath="/Users/daniel/Documents/cs336/homework/assignment1-basics/output.nosync/vocab.json",
        merges_filepath="/Users/daniel/Documents/cs336/homework/assignment1-basics/output.nosync/merges.json",
    )

    deserialized_vocab, deserialized_merges = deserialize_vocab_and_merges(
        vocab_filepath="/Users/daniel/Documents/cs336/homework/assignment1-basics/output.nosync/vocab.json",
        merges_filepath="/Users/daniel/Documents/cs336/homework/assignment1-basics/output.nosync/merges.json",
    )

    # print last 10 vocab items
    print("last 10 vocab items:")
    for token_id in range(len(deserialized_vocab) - 10, len(deserialized_vocab)):
        print(f"{token_id}: {deserialized_vocab[token_id]}")
    # print top 10 merges
    print("\nTop 10 merges:")
    for i in range(10):
        print(f"{i}: {deserialized_merges[i]}")
