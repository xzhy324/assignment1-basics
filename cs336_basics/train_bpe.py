import os
from concurrent.futures import ProcessPoolExecutor
import regex as re
from cs336_basics.pretokenization import find_chunk_boundaries



def _count_word_like_pieces(args: tuple[str, int, int,list[bytes]] ) -> dict[tuple[bytes, ...], int]:
    input_path, start, end, special_tokens_bytes = args
    chunk_word_like_pieces: dict[tuple[bytes, ...], int] = {}
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # 先按照special tokens将大chunk直接切分成多个小chunk
    special_tokens_pattern = b"|".join(re.escape(token) for token in special_tokens_bytes).decode("utf-8")
    smaller_chunks = re.split(
        special_tokens_pattern,
        chunk
    )

    # 对每个小chunk，按照PATTERN切分成word-like pieces，并统计这些piece的频次
    piece_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" # GPT2的pretokenizer使用的pattern
    for smaller_chunk in smaller_chunks:
        for match in re.finditer(piece_pattern, smaller_chunk):
            piece = match.group(0).encode("utf-8")
            byte_tokens = tuple(piece[i:i+1] for i in range(len(piece))) # 将piece切分成一个个byte token，组成一个tuple作为key
            chunk_word_like_pieces[byte_tokens] = chunk_word_like_pieces.get(byte_tokens, 0) + 1
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
    num_processes = max(1, os.cpu_count() - 1) # leave 1 CPU free
    boundaries = find_chunk_boundaries(f, desired_num_chunks=num_processes, split_special_token=special_tokens_bytes[0])

    word_like_pieces:list[dict[tuple[bytes, ...], int]] = [] # one per chunk
    
    # 并行统计每个 chunk 里的 word-like piece 频次
    chunk_ranges = list(zip(boundaries[:-1], boundaries[1:]))
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for chunk_word_like_pieces in executor.map(
            _count_word_like_pieces,
            ((str(input_path), start, end, special_tokens_bytes) for start, end in chunk_ranges),
        ):
            word_like_pieces.append(chunk_word_like_pieces)
    
    # 合并所有chunk的统计结果
    total_word_like_pieces: dict[tuple[bytes, ...], int] = {}
    for chunk_word_like_piece in word_like_pieces:
        for piece, count in chunk_word_like_piece.items():
            total_word_like_pieces[piece] = total_word_like_pieces.get(piece, 0) + count
    
    # # DEBUG print
    # word_like_pieces_sorted = sorted(total_word_like_pieces.items(), key=lambda x: x[1], reverse=True)
    # print(f"Total unique word-like pieces: {len(total_word_like_pieces)}")
    # # DEBUG print top 10
    # for piece, count in word_like_pieces_sorted[:10]:
    #     print(f"{piece}: {count}")

    all_pairs: dict[tuple[bytes, bytes], int] = {} # (bytes, bytes) -> count
    for piece, count in total_word_like_pieces.items():
        for i in range(len(piece) - 1):
            pair = (piece[i], piece[i+1])
            all_pairs[pair] = all_pairs.get(pair, 0) + count
    # # DEBUG print top 10 pairs
    # all_pairs_sorted = sorted(all_pairs.items(), key=lambda x: (x[1], x[0]), reverse=True)
    # print(f"Total unique pairs: {len(all_pairs)}")
    # for pair, count in all_pairs_sorted[:10]:
    #     print(f"{pair}: {count}")
    
    # 添加最频繁的 pairs 到 vocab 和 merges，直到达到 vocab_size
    while len(vocab) < vocab_size:
        # 从all pairs中取出当前最频繁的 pair
        pair, count = max(all_pairs.items(), key=lambda x: (x[1],x[0])) # 先按频次排序，频次相同则按pair的字典序排序
        # 更新merges和vocab
        merges.append(pair)
        new_token = pair[0] + pair[1]
        vocab[len(vocab)] = new_token
        # 构造一个新的total_word_like_pieces，包含所有替换了当前选中的最大pair的piece
        updated_word_like_pieces: dict[tuple[bytes, ...], int] = {}
        for piece, piece_count in total_word_like_pieces.items():
            new_piece = []
            i=0 # 这个循环的目的是将当前的piece中所有当前选中的最大pair（如果有的话）替换成new_token，生成一个新的piece
            while i < len(piece):
                if i < len(piece) - 1 and (piece[i], piece[i+1]) == pair:
                    new_piece.append(new_token)
                    i += 2
                else:
                    new_piece.append(piece[i])
                    i += 1
            new_piece_tuple = tuple(new_piece)
            updated_word_like_pieces[new_piece_tuple] = updated_word_like_pieces.get(new_piece_tuple, 0) + piece_count
        
        # 全量重算all_pairs
        all_pairs.clear()
        for piece, count in updated_word_like_pieces.items():
            for i in range(len(piece) - 1):
                pair = (piece[i], piece[i+1])
                all_pairs[pair] = all_pairs.get(pair, 0) + count
        # 更新total_word_like_pieces
        total_word_like_pieces = updated_word_like_pieces
            
        
            
    return vocab, merges


if __name__ == "__main__":
    rvocab, rmerges = train_bpe(
        input_path="/Users/daniel/Documents/cs336/homework/assignment1-basics/data.nosync/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    
    print("Vocabulary:")
    for token_id, token_bytes in rvocab.items():
        print(f"{token_id}: {token_bytes}")
    print("\nMerges:")
    for merge in rmerges:
        print(merge)

