from collections import Counter
from typing import Callable, Optional, Union, Any, Sequence

import numpy as np
from ignite.metrics import Accuracy, Precision, Recall, MeanAbsoluteError, Metric, Loss
from ignite.metrics.metric import reinit__is_reduced
from torch import nn

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


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


def _binarize_sigmoid(y_pred):
    y_pred = nn.Sigmoid()(y_pred)
    y_pred = torch.ge(y_pred, 0.5).int()
    return y_pred


def binarize_softmax(output):
    y_pred, y = output['y_pred'], output['y']
    sm = nn.Softmax(dim=1)(y_pred)
    _, i = torch.max(sm, dim=1)
    # y_pred = i

    y_pred = torch.nn.functional.one_hot(i, sm.shape[1])
    # y = torch.nn.functional.one_hot(y,sm.shape[1])
    return y_pred, y


def binarize_sigmoid(output):
    y_pred, y = output['y_pred'], output['y']
    y_pred = _binarize_sigmoid(y_pred)
    return (y_pred, y)


class ErrorIndices(Metric):

    def __init__(self, output_transform: Callable = lambda x: x, device: Optional[Union[str, torch.device]] = None):
        super().__init__(output_transform, device)
        self.batch_counter = 0

    def reset(self) -> None:
        self.batch_counter = 0

    def update(self, output) -> None:
        y_pred, y = output['y_pred'], output['y']
        wrong = y_pred != y
        self.batch_counter = self.batch_counter + len(y)

    def compute(self) -> Any:
        pass


class Minimum(SimpleAccumulator):
    def compute(self) -> Any:
        return np.min(self.values)


class Maximum(SimpleAccumulator):
    def compute(self) -> Any:
        return np.max(self.values)


class HistPlotter(SimpleAccumulator):
    def __init__(self, visdom_logger, output_transform: Callable = lambda x: x,
                 device: Optional[Union[str, torch.device]] = None):
        super().__init__(output_transform, device)
        self.visdom_logger = visdom_logger

    def compute(self) -> Any:
        self.visdom_logger.histogram(self.values)
        return np.mean(self.values)


class SimpleAggregate(Metric):
    def __init__(self, base_metric, aggregate, output_transform: Callable = lambda x: x,
                 device: Optional[Union[str, torch.device]] = None):
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


class CustomLossWrapper(Loss):

    @reinit__is_reduced
    def update(self, output: Sequence[Union[torch.Tensor, dict]]) -> None:
        if len(output) == 2:
            y_pred, y = output
            kwargs = {}
        else:
            y_pred, y, kwargs = output
        average_loss = self._loss_fn(y_pred, y, **kwargs)

        # empty batches cause nan losses, so drop them
        if y.nelement() == 0:
            return

        if len(average_loss.shape) != 0:
            raise ValueError("loss_fn did not return the average loss.")

        N = self._batch_size(y)
        self._sum += average_loss.item() * N
        self._num_examples += N

loss_mapping = {'binary': BCEWithLogitsLoss(),'multi': CrossEntropyLoss(),
                'regression': MSELoss()}


default_metric_mapping = {
    'regression': [('mae', MeanAbsoluteError, None, {}),
                   ('mean', Average, lambda output: output['y_pred'], {}),
                   ('var', Variance, lambda output: output['y_pred'], {})],

    'binary': [('accuracy', Accuracy, binarize_sigmoid, {}),
               ('p', Precision, binarize_sigmoid, {}),
               ('r', Recall, binarize_sigmoid, {}),
               ('pos_share', PositiveShare, binarize_sigmoid, {})],

    'multi': [
        ('accuracy', Accuracy, binarize_softmax, {}),
        ('ap', Precision, binarize_softmax, {'average': True}),
        ('ar', Recall, binarize_softmax, {'average': True})
    ]

}


# extracts the subdictionary for prediction and target, leaves the rest unaltered
def map_output_dict(output_dict, attribute_name):
    new_dict = {}
    if attribute_name:
        tgt = output_dict['y'][attribute_name]
        pred = output_dict['y_pred'][attribute_name]
    else:
        tgt = output_dict['y']
        pred = output_dict['y_pred']

    # make sure there's no nan values
    to_keep = ~torch.isnan(tgt)
    output_dict['y'][attribute_name] = tgt[to_keep]
    output_dict['y_pred'][attribute_name] = pred[to_keep]

    for k, v in output_dict.items():
        if k == 'y' or k == 'y_pred':
            new_dict[k] = output_dict[k][attribute_name]
        else:
            new_dict[k] = v
    return new_dict


def map_output_dict_to_tuple(output_dict, attribute_name):
    if attribute_name:
        tgt = output_dict['y'][attribute_name]
        pred = output_dict['y_pred'][attribute_name]
    else:
        tgt = output_dict['y']
        pred = output_dict['y_pred']
    to_keep = ~torch.isnan(tgt)
    return pred[to_keep], tgt[to_keep]


def make_loss_extractor(attribute_name):
    return lambda x: map_output_dict_to_tuple(x, attribute_name)


def make_metric_extractor(attribute_name):
    return lambda x: map_output_dict(x, attribute_name=attribute_name)


def build_transform(map_fn, transform_fn):
    return lambda x: transform_fn(map_fn(x))


def obtain_metrics(attribute_specs, extract_multiple_atts=False, metric_mapping=None,
                   add_loss=True):
    if metric_mapping == None:
        metric_mapping = default_metric_mapping

    # should return only one, but specified multiple --> error
    # should return multiple, but specified only one --> no error
    if ~extract_multiple_atts & (len(attribute_specs) > 1):
        raise ValueError(f'Was asked to extract single attribute, but found {len(attribute_specs)}')

    final_metrics = {}
    for att_spec in attribute_specs:
        if add_loss:
            # make loss metric
            loss_name = 'loss_' + att_spec.name
            loss_fn = loss_mapping[att_spec.target_type]

            if extract_multiple_atts:
                transform = make_loss_extractor(att_spec.name)
            else:
                transform = make_loss_extractor(None)
            loss_to_add = CustomLossWrapper(loss_fn, output_transform=transform)
            final_metrics[loss_name] = loss_to_add

        # based on the target type, add metrics
        metric_extractor = make_metric_extractor(att_spec.name)
        metric_comps = metric_mapping[att_spec.target_type]
        for comp in metric_comps:

            metric_name, metric_fn, transform_fn, invoke_kwargs = comp
            if extract_multiple_atts:
                final_transform = metric_extractor
                if transform_fn and callable(transform_fn):
                    final_transform = build_transform(metric_extractor, transform_fn)
            else:
                if transform_fn:
                    final_transform = transform_fn
                else:
                    final_transform = lambda x: x

            invoke_fn = metric_fn(output_transform=final_transform, **invoke_kwargs)
            metric_name = metric_name + '_' + att_spec.name
            final_metrics[metric_name] = invoke_fn
    return final_metrics