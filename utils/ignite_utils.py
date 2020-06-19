from collections import Counter
from functools import reduce

import numpy as np
from typing import Any, Callable, Optional, Union

import torch
from ignite.metrics import Metric
from torch import nn

def pytorch_count_params(model):
  "count number trainable parameters in a pytorch model"
  total_params = sum(reduce( lambda a, b: a*b, x.size()) for x in model.parameters())
  return total_params

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


def _binarize_softmax(y_pred):
    sm = nn.Softmax(dim=1)(y_pred)
    _, i = torch.max(sm, dim=1)
    return i

def binarize_softmax(output):
    y_pred, y = output['y_pred'], output['y']
    sm = nn.Softmax(dim=1)(y_pred)
    _, i = torch.max(sm, dim=1)
    # y_pred = i

    y_pred = torch.nn.functional.one_hot(i,sm.shape[1])
   # y = torch.nn.functional.one_hot(y,sm.shape[1])
    return y_pred, y

def _binarize_sigmoid(y_pred):
    y_pred = nn.Sigmoid()(y_pred)
    y_pred = torch.ge(y_pred, 0.5).int()
    return y_pred

def binarize_sigmoid(output):
    y_pred, y = output['y_pred'], output['y']
    y_pred = _binarize_sigmoid(y_pred)
    return y_pred, y