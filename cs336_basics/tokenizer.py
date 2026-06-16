"""Tokenizer implementation scaffolding for CS336 basics assignment."""

from collections.abc import Iterable, Iterator

from cs336_basics.train_bpe import deserialize_vocab_and_merges
import regex as re
import os


class Tokenizer:
    """A tokenizer interface for loading, encoding, and decoding text."""

    def __init__(self, vocab, merges, special_tokens=None):
        """Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.

        Args:
            vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
            special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
                be split into multiple tokens, and will always be kept as a single token.
        """
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.bytes2id = {
            v: k for k, v in self.vocab.items()
        }  # vocab的反向映射，方便根据token bytes找到token id
        self.merges_rank = {
            merge: rank for rank, merge in enumerate(self.merges)
        }  # merges的反向映射，方便根据merge找到rank（越小的rank说明这个merge越早被创建，优先级越高）

    @classmethod
    def from_files(
        cls, vocab_filepath, merges_filepath, special_tokens=None
    ) -> "Tokenizer":
        """Construct a tokenizer from serialized vocabulary and merges files.

        Args:
            vocab_filepath (str): Path to the serialized vocabulary file.
            merges_filepath (str): Path to the serialized merges file.
            special_tokens (list[str] | None): A list of string special tokens for the tokenizer.

        Returns:
            Tokenizer: A constructed Tokenizer instance.
        """
        vocab, merges = deserialize_vocab_and_merges(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs.

        Args:
            text (str): The input text to encode.

        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        ids = []
        # 先按照special tokens切分成多个小chunk , special token也是一个chunk
        if self.special_tokens:
            # 这里特别注意，special tokens里面可能overlapping（例如：["<|endoftext|>", "<|endoftext|><|endoftext|>"]），所以我们需要先按照长度对special tokens进行降序排序，保证长的special token优先被匹配到
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            text_chunks = re.split(
                f"({'|'.join(map(re.escape, sorted_special_tokens))})", text
            )
        else:  # corner case: special_tokens = []
            text_chunks = [text]
        for chunk in text_chunks:
            if chunk in self.special_tokens:
                # 如果是special token，直接转换成token id
                token_id = self.bytes2id.get(chunk.encode("utf-8"), None)
                assert (
                    token_id is not None
                ), f"Special token '{chunk}' not found in vocab, please check your special tokens and vocab consistency."
                ids.append(token_id)
            else:
                # 否则继续按照正则表达式切分成更小的pre-tokens
                # 例如："abc def" => ["abc", "def"]
                gpt2_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                pre_tokens = re.finditer(gpt2_pattern, chunk)
                for pre_token in pre_tokens:
                    pre_token = pre_token.group(0).encode(
                        "utf-8"
                    )  # 将pre-token从str转换成bytes
                    appending_bytes = [b.to_bytes() for b in pre_token]

                    while True:
                        best_merge_rank = None
                        # 确定当前这个pre-token piece的best merge rank
                        for index in range(len(appending_bytes) - 1):
                            pair = (appending_bytes[index], appending_bytes[index + 1])
                            # 遍历所有的pair，看看哪个pair在merges里优先级最高（rank最小），把那个pair merge掉
                            # （注意这个pair可能出现了多次，需要从左到右多次处理piece，直到不存在可以应用当前这个merge的pair
                            current_merge_rank = self.merges_rank.get(pair, None)
                            if current_merge_rank is not None:
                                if (
                                    best_merge_rank is None
                                    or current_merge_rank < best_merge_rank
                                ):
                                    best_merge_rank = current_merge_rank
                        # 如果best_merge_rank没有被更新过，说明这个piece没有任何合法的merge了，我们就可以停止寻找merge了
                        if best_merge_rank is None:
                            break
                        # 这里我们拿到了best_merge_rank，接下来我们需要把这个rank对应的merge应用到当前的piece上
                        # 从左到右贪心地把这个merge应用到当前的piece上，生成一个new_appending_bytes，结束后把new_appending_bytes赋值给appending_bytes
                        best_merge = self.merges[best_merge_rank]
                        new_appending_bytes = []
                        index = 0
                        while index < len(appending_bytes):
                            if (
                                index < len(appending_bytes) - 1
                                and (
                                    appending_bytes[index],
                                    appending_bytes[index + 1],
                                )
                                == best_merge
                            ):
                                # 把这个pair merge掉
                                new_appending_bytes.append(
                                    best_merge[0] + best_merge[1]
                                )
                                index += 2  # 跳过下一个，因为它已经被merge掉了
                            else:
                                new_appending_bytes.append(appending_bytes[index])
                                index += 1
                        appending_bytes = new_appending_bytes

                    for token_bytes in appending_bytes:
                        token_id = self.bytes2id.get(token_bytes, None)
                        assert (
                            token_id is not None
                        ), f"Token bytes {token_bytes} not found in vocab，something went wrong during encoding."
                        ids.append(token_id)
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into
        memory."""
        for txt in iterable:
            yield from self.encode(txt)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text.

        Args:
            ids (list[int]): A list of token IDs to decode.

        Returns:
            str: The decoded text.
        """
        output_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return output_bytes.decode("utf-8", errors="replace")


if __name__ == "__main__":
    # Example usage of the Tokenizer class
    THIS_FILEPATH = os.path.abspath(__file__)
    VOCAB_FILEPATH = os.path.join(os.path.dirname(THIS_FILEPATH), "../output/TinyStoriesV2-GPT4-train.txt_vocab.json")
    MERGES_FILEPATH = os.path.join(os.path.dirname(THIS_FILEPATH), "../output/TinyStoriesV2-GPT4-train.txt_merges.json")

    TEST_TEXT = "Once upon a time, there was a little cat named Whiskers. <|endoftext|>"

    rvocab, rmerges = deserialize_vocab_and_merges(
        vocab_filepath=VOCAB_FILEPATH,
        merges_filepath=MERGES_FILEPATH,
    )

    tokenizer = Tokenizer(
        vocab=rvocab,
        merges=rmerges,
        special_tokens=["<|endoftext|>"],
    )

    # Example encoding and decoding (these will raise NotImplementedError until implemented)
    encoded = tokenizer.encode(TEST_TEXT)
    print(encoded)
    decoded = tokenizer.decode(encoded)
    print(decoded)
    assert TEST_TEXT == decoded, "Decoded text does not match original text!"

    compression_ratio = len(TEST_TEXT.encode("utf-8")) / len(encoded)
    print(f"Compression ratio: {compression_ratio:.2f}")
