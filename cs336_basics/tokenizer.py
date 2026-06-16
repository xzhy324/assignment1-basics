"""Tokenizer implementation scaffolding for CS336 basics assignment."""

from collections.abc import Iterable, Iterator

from cs336_basics.train_bpe import deserialize_vocab_and_merges


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
        self.merges = merges
        self.special_tokens = special_tokens or []

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
        # step1: pre-tokenization

        # step2: apply merges to get final tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into
        memory."""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text.

        Args:
            ids (list[int]): A list of token IDs to decode.

        Returns:
            str: The decoded text.
        """
        # Implementation for decoding token IDs into text would go here
        raise NotImplementedError("Decoding is not implemented yet.")


if __name__ == "__main__":
    # Example usage of the Tokenizer class
    rvocab, rmerges = deserialize_vocab_and_merges(
        vocab_filepath="/Users/daniel/Documents/cs336/homework/assignment1-basics/output.nosync/vocab.json",
        merges_filepath="/Users/daniel/Documents/cs336/homework/assignment1-basics/output.nosync/merges.json",
    )

    tokenizer = Tokenizer(
        vocab=rvocab,
        merges=rmerges,
        special_tokens=["<|endoftext|>"],
    )

    # Example encoding and decoding (these will raise NotImplementedError until implemented)
    encoded = tokenizer.encode("Hello, world!")
    print(encoded)
    decoded = tokenizer.decode(encoded)
    print(decoded)
