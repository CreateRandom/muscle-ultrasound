from typing import Any, Callable, Optional, Union, Sequence, Tuple, Dict

import torch

from ignite.engine import _prepare_batch, Engine, DeterministicEngine
from ignite.metrics import Metric

from utils.metric_utils import loss_mapping


def create_bag_attention_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    att_specs,
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

        loss_dict = {}
        for att_spec in att_specs:
            name = att_spec.name
            loss_fn = loss_mapping[att_spec.target_type]
            # filter nan values in y
            to_keep = ~torch.isnan(y[name])
            _y = y[name][to_keep]
            _y_pred = y_pred[name][to_keep]
            # some batches might be entirely nan, skip these
            if _y.nelement() > 0:
                loss = loss_fn(_y_pred, _y)
                loss_dict[name] = loss

        # TODO parametrize
        loss_weights = {'Age': 0.001, 'BMI': 0.001}

        weighted_losses = dict([(k, loss_weights[k] * v) if k in loss_weights else (k,v) for k, v in loss_dict.items() ])

        loss = sum(weighted_losses.values())
        if loss > 0:
            loss.backward()

            if on_tpu:
                xm.optimizer_step(optimizer, barrier=True)
            else:
                optimizer.step()
        else:
            loss = torch.tensor(0)

        # flatten the scores
        att_scores_flat = flatten_attention_scores(attention_outputs)

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

def create_bag_attention_evaluator(
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