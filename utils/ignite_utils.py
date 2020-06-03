from collections import Counter
import numpy as np
from typing import Any, Callable, Optional, Union

import torch
from ignite.metrics import Metric

class SimpleAccumulator(Metric):
    def __init__(self, output_transform: Callable = lambda x: x, device: Optional[Union[str, torch.device]] = None):
        super().__init__(output_transform, device)
        self.values = []

    def reset(self) -> None:
        self.values = []

    def update(self, output) -> None:
        self.values.extend(list(output.flatten().cpu().numpy()))

    def compute(self) -> Any:
        pass

class PositiveShare(Metric):

    def __init__(self, output_transform: Callable = lambda x: x, device: Optional[Union[str, torch.device]] = None):
        super().__init__(output_transform, device)
        self.counter = Counter()

    def reset(self) -> None:
        self.counter = Counter()

    def update(self, output) -> None:
        y_pred, y = output
        self.counter.update(list(y_pred.flatten().cpu().numpy()))

    def compute(self) -> Any:
        total_count = sum(self.counter.values())
        positive_count = self.counter[1]
        return float(positive_count / total_count)

class Variance(SimpleAccumulator):
    def compute(self) -> Any:
        return np.var(self.values)

class Average(SimpleAccumulator):
    def compute(self) -> Any:
        return np.mean(self.values)