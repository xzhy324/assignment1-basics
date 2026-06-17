from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class BPETrainConfig:
    input_path: Path
    output_dir: Path
    vocab_size: int
    special_tokens: tuple[str, ...]
    num_workers: int
    num_chunks: int
    log_memory: bool


def load_config() -> BPETrainConfig:
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = os.environ.get("BPE_DATA_RELATIVE_DIR", str(base_dir / "data/TinyStoriesV2-GPT4-train.txt"))
    default_input = base_dir / data_dir
    
    cpu_count = os.cpu_count() or 2
    num_workers = int(os.environ.get("BPE_NUM_PROCESSES", str(cpu_count)))
    return BPETrainConfig(
        input_path=default_input.resolve(),
        output_dir=(base_dir / "output").resolve(),
        vocab_size=int(os.environ.get("BPE_VOCAB_SIZE", "10000")),
        special_tokens=tuple(
            os.environ.get("BPE_SPECIAL_TOKENS", "<|endoftext|>").split(",")
        ),
        num_workers=num_workers,
        num_chunks=int(os.environ.get("BPE_NUM_CHUNKS", str(num_workers * 16))),
        log_memory=os.environ.get("BPE_LOG_MEMORY", "0") == "1",
    )