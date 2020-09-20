from typing import Any, Callable, Optional, Union, Sequence, Tuple, Dict

import torch

from ignite.engine import _prepare_batch, Engine, DeterministicEngine
from ignite.metrics import Metric

from utils.coral import coral
from utils.ignite_metrics import loss_mapping

from utils.utils import geometric_mean


def compute_classification_loss(head_preds, y, att_specs, loss_weights=None, loss_kwargs_mapping=None,
                                use_gmean=False):
    if not loss_weights:
        loss_weights = {}
    loss_dict = {}
    for att_spec in att_specs:
        if loss_kwargs_mapping and att_spec.name in loss_kwargs_mapping:
            loss_kwargs = loss_kwargs_mapping[att_spec.name]
        else:
            loss_kwargs = {}

        name = att_spec.name
        loss_fn = loss_mapping[att_spec.target_type]
        loss_fn = loss_fn(**loss_kwargs)
        # filter nan values in y
        to_keep = ~torch.isnan(y[name])
        _y = y[name][to_keep]
        _y_pred = head_preds[name][to_keep]
        # some batches might be entirely nan, skip these
        if _y.nelement() > 0:
            loss = loss_fn(_y_pred, _y)
            loss_dict[name] = loss
    # side-step the computation if there is only one elem
    if len(loss_dict) == 1:
        loss = list(loss_dict.values())[0]
    else:
        # apply weights
        weighted_losses = dict([(k, loss_weights[k] * v) if k in loss_weights else (k, v) for k, v in loss_dict.items()])
        # use the geometric mean of the losses
        loss_values = torch.stack(tuple(weighted_losses.values()))
        if use_gmean:
            loss = geometric_mean(loss_values,dim=0)
        else:
            loss = torch.mean(loss_values,dim=0)

    return loss

def compute_coral_loss(src_head_acts, tgt_head_acts, att_specs, layer_inds, loss_weights):
    loss_dict = {}
    for att_spec in att_specs:
        name = att_spec.name
        c_loss = 0
        for layer in layer_inds:
            c_loss = c_loss + coral(src_head_acts[name][layer], tgt_head_acts[name][layer])

        loss_dict[name] = c_loss

    weighted_losses = dict([(k, loss_weights[k] * v) if k in loss_weights else (k, v) for k, v in loss_dict.items()])

    loss = sum(weighted_losses.values())

    return loss, loss_dict

def create_bag_attention_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    att_specs,
    att_loss_weights = None,
    loss_kwargs_mapping=None,
    use_gmean=False,
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

    if not att_loss_weights:
        att_loss_weights = {}

    if on_tpu:
        try:
            import torch_xla.core.xla_model as xm
        except ImportError:
            raise RuntimeError("In order to run on TPU, please install PyTorch XLA")

    def _update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        model_return_dict = model(x)
        head_preds = model_return_dict['head_preds']
        loss = compute_classification_loss(head_preds, y, att_specs, att_loss_weights, loss_kwargs_mapping,
                                           use_gmean=use_gmean)

        if loss > 0:
            loss.backward()

            if on_tpu:
                xm.optimizer_step(optimizer, barrier=True)
            else:
                optimizer.step()
        else:
            loss = torch.tensor(0)
        return_dict = {}
        return_dict['preds'] = head_preds

        if 'attention_outputs' in model_return_dict:
            # flatten the scores
            att_scores_flat = flatten_attention_scores(model_return_dict['attention_outputs'])
            return_dict['atts'] = att_scores_flat

        return output_transform(x, y, return_dict, loss)

    trainer = Engine(_update) if not deterministic else DeterministicEngine(_update)

    return trainer


def flatten_attention_scores(attention_outputs):
    att_scores_flat = []

    if attention_outputs:
        for x in attention_outputs: att_scores_flat.append(x.flatten())
        att_scores_flat = torch.cat(att_scores_flat)
    return att_scores_flat

def create_image_baseline_evaluator(
    model: torch.nn.Module,
    metrics: Optional[Dict[str, Metric]] = None,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = lambda x, y, y_pred: (y_pred, y),
) -> Engine:

    metrics = metrics or {}

    def _inference(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()
        with torch.no_grad():
            pred_output = {}
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            model_return_dict = model(x)
            pred_output['preds'] = model_return_dict['preds']

            # if image wise predications are available, flatten and return them
            if 'image_preds' in model_return_dict:
                image_preds_flat = flatten_attention_scores(model_return_dict['image_preds'])
                pred_output['imagewise_preds'] = image_preds_flat
            return output_transform(x, y, pred_output)

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator

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
            pred_output = {}
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            model_return_dict = model(x)
            pred_output['preds'] = model_return_dict['head_preds']

            # if attention scores are available, flatten and return them
            if 'attention_outputs' in model_return_dict:
                att_scores_flat = flatten_attention_scores(model_return_dict['attention_outputs'])
                pred_output['atts'] = att_scores_flat
            return output_transform(x, y, pred_output)

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator

def create_da_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    att_specs,
    lambda_weight,
    layers_to_compute_da_on,
    loss_kwargs_mapping = None,
    att_loss_weights = None,
    use_gmean=False,
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

    if not att_loss_weights:
        att_loss_weights = {}

    if on_tpu:
        try:
            import torch_xla.core.xla_model as xm
        except ImportError:
            raise RuntimeError("In order to run on TPU, please install PyTorch XLA")

    def _update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        pred_output = {}
        model.train()
        optimizer.zero_grad()
        source, target = prepare_batch(batch, device=device, non_blocking=non_blocking)
        x_src, y_src = source
        x_tgt, _ = target
        model_out_src = model(x_src)
        head_preds = model_out_src['head_preds']
        class_loss = compute_classification_loss(head_preds, y_src, att_specs, att_loss_weights,loss_kwargs_mapping,
                                                 use_gmean)

        # store the predictions
        pred_output['preds'] = model_out_src['head_preds']

        # if attention scores are available, flatten and return them
        if 'attention_outputs' in model_out_src:
            att_scores_flat = flatten_attention_scores(model_out_src['attention_outputs'])
            pred_output['atts'] = att_scores_flat

        # predict the target data
        model_out_tgt = model(x_tgt)


        # get the layer_act_dict
        coral_loss, coral_loss_dict = compute_coral_loss(model_out_src['head_acts'], model_out_tgt['head_acts'], att_specs,
                                        layer_inds=layers_to_compute_da_on, loss_weights={})
        loss = class_loss + lambda_weight * coral_loss
        loss.backward()

        if on_tpu:
            xm.optimizer_step(optimizer, barrier=True)
        else:
            optimizer.step()

        pred_output['coral_losses'] = coral_loss_dict

        return output_transform(x_src, y_src, pred_output, loss)

    trainer = Engine(_update) if not deterministic else DeterministicEngine(_update)

    return trainer

class StateDictWrapper(object):
    def __init__(self, object):
        self.object = object
    def state_dict(self):
        return self.object