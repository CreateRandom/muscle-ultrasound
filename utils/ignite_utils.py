from collections import Counter
from typing import Any, Callable, Optional, Union

import torch
from ignite.metrics import Metric


class ValueCount(Metric):

    def __init__(self, output_transform: Callable = lambda x: x, device: Optional[Union[str, torch.device]] = None):
        super().__init__(output_transform, device)
        self.counter = Counter()

    def reset(self) -> None:
        self.counter = Counter()

    def update(self, output) -> None:
        y_pred, y = output
        self._check_shape((y_pred, y))
        self._check_type((y_pred, y))
        self.counter.update(list(y_pred))

    def compute(self) -> Any:
        counts = self.counter.most_common()
        return counts