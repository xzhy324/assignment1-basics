from collections.abc import Callable, Iterable
from typing import Optional

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self):
        pass

    def step(self, closure: Optional[Callable] = None):
        pass
s