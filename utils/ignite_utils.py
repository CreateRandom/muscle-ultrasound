from collections import Counter
from functools import reduce

import numpy as np
from typing import Any, Callable, Optional, Union, Sequence, Tuple, Dict

import torch

from ignite.engine import _prepare_batch, Engine, DeterministicEngine
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

class Minimum(SimpleAccumulator):
    def compute(self) -> Any:
        return np.min(self.values)

class Maximum(SimpleAccumulator):
    def compute(self) -> Any:
        return np.max(self.values)

class HistPlotter(SimpleAccumulator):
    def __init__(self, visdom_logger, output_transform: Callable = lambda x: x, device: Optional[Union[str, torch.device]] = None):
        super().__init__(output_transform, device)
        self.visdom_logger = visdom_logger
    def compute(self) -> Any:
        self.visdom_logger.histogram(self.values)
        return np.mean(self.values)

class SimpleAggregate(Metric):
    def __init__(self, base_metric, aggregate, output_transform: Callable = lambda x: x, device: Optional[Union[str, torch.device]] = None):
        super().__init__(output_transform, device)
        self.base_metric = base_metric
        self.aggregate = aggregate
        self.values = []

    def reset(self) -> None:
        # self.base_metric.reset()
        #     self.values = []
        pass
    def update(self, output) -> None:
        self.base_metric.update(output)

    def compute(self) -> Any:
        self.values.append(self.base_metric.compute())
        return self.aggregate(self.values)


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

def create_custom_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = lambda x, y, y_pred, loss: loss.item(),
    deterministic: bool = False,
) -> Engine:
    """
    Factory function for creating a trainer for supervised models.
    Args:
        model (`torch.nn.Module`): the model to train.
        optimizer (`torch.optim.Optimizer`): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        device (str, optional): device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
            Device can be CPU, GPU or TPU.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.
        deterministic (bool, optional): if True, returns deterministic engine of type
            :class:`~ignite.engine.deterministic.DeterministicEngine`, otherwise :class:`~ignite.engine.Engine`
            (default: False).
    Note:
        `engine.state.output` for this engine is defined by `output_transform` parameter and is the loss
        of the processed batch by default.
    .. warning::
        The internal use of `device` has changed.
        `device` will now *only* be used to move the input data to the correct device.
        The `model` should be moved by the user before creating an optimizer.
        For more information see:
        * `PyTorch Documentation <https://pytorch.org/docs/stable/optim.html#constructing-it>`_
        * `PyTorch's Explanation <https://github.com/pytorch/pytorch/issues/7844#issuecomment-503713840>`_
    Returns:
        Engine: a trainer engine with supervised update function.
    """

    device_type = device.type if isinstance(device, torch.device) else device
    on_tpu = "xla" in device_type if device_type is not None else False

    if on_tpu:
        try:
            import torch_xla.core.xla_model as xm
        except ImportError:
            raise RuntimeError("In order to run on TPU, please install PyTorch XLA")

    def _update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred, attention_outputs = model(x)
        # flatten the scores
        att_scores_flat = flatten_attention_scores(attention_outputs)

        loss = loss_fn(y_pred, y)
        loss.backward()

        if on_tpu:
            xm.optimizer_step(optimizer, barrier=True)
        else:
            optimizer.step()

        y_pred = {'preds': y_pred, 'atts': att_scores_flat}
        return output_transform(x, y, y_pred, loss)

    trainer = Engine(_update) if not deterministic else DeterministicEngine(_update)

    return trainer

def flatten_attention_scores(attention_outputs):
    att_scores_flat = []

    if attention_outputs:
        for x in attention_outputs: att_scores_flat.append(x.flatten())
        att_scores_flat = torch.cat(att_scores_flat)
    return att_scores_flat

def create_custom_evaluator(
    model: torch.nn.Module,
    metrics: Optional[Dict[str, Metric]] = None,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = lambda x, y, y_pred: (y_pred, y),
) -> Engine:
    """
    Factory function for creating an evaluator for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        metrics (dict of str - :class:`~ignite.metrics.Metric`): a map of metric names to Metrics.
        device (str, optional): device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics.

    Note:
        `engine.state.output` for this engine is defind by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    .. warning::

        The internal use of `device` has changed.
        `device` will now *only* be used to move the input data to the correct device.
        The `model` should be moved by the user before creating an optimizer.

        For more information see:

        * `PyTorch Documentation <https://pytorch.org/docs/stable/optim.html#constructing-it>`_
        * `PyTorch's Explanation <https://github.com/pytorch/pytorch/issues/7844#issuecomment-503713840>`_

    Returns:
        Engine: an evaluator engine with supervised inference function.
    """
    metrics = metrics or {}

    def _inference(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred, attention_outputs = model(x)
            # flatten the scores
            att_scores_flat = flatten_attention_scores(attention_outputs)
            y_pred = {'preds': y_pred, 'atts': att_scores_flat}
            return output_transform(x, y, y_pred)

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator
