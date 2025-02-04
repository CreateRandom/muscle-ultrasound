from collections import Counter
from typing import Callable, Optional, Union, Any, Sequence

import numpy as np
import torch
from ignite.metrics import Metric, Loss, MeanAbsoluteError, Accuracy, Precision, Recall
from ignite.metrics.metric import reinit__is_reduced
from sklearn.metrics import roc_auc_score, roc_curve
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from utils.binarize_utils import binarize_sigmoid, binarize_softmax, apply_sigmoid


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

class AUC(Metric):
    def __init__(self, output_transform: Callable = lambda x: x, device: Optional[Union[str, torch.device]] = None):
        super().__init__(output_transform, device)
        self.ys = []
        self.ypreds = []

    def reset(self) -> None:
        self.ys = []
        self.ypreds = []

    def update(self, output) -> None:
        y_pred, y = output
        self.ys.extend(list(y.flatten().cpu().numpy()))
        self.ypreds.extend(list(y_pred.flatten().cpu().numpy()))

    def compute(self) -> Any:
        return roc_auc_score(self.ys, self.ypreds)

class YoudensJ(AUC):

    def compute(self) -> Any:
        fpr, tpr, thresholds = roc_curve(self.ys, self.ypreds)
        J = tpr - fpr
        best_ind = np.argmax(J)
        best_threshold = thresholds[best_ind]
        return best_threshold


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


loss_mapping = {'binary': BCEWithLogitsLoss,'multi': CrossEntropyLoss,
                'regression': MSELoss}
default_metric_mapping = {
    'regression': [('mae', MeanAbsoluteError, None, {}),
                   ('mean', Average, lambda output: output['y_pred'], {}),
                   ('var', Variance, lambda output: output['y_pred'], {})],

    'binary': [('accuracy', Accuracy, binarize_sigmoid, {}),
               ('p', Precision, binarize_sigmoid, {}),
               ('r', Recall, binarize_sigmoid, {}),
               ('pos_share', PositiveShare, binarize_sigmoid, {}),
               ('auc', AUC, apply_sigmoid,{}),
               ('best_threshold', YoudensJ, apply_sigmoid,{}),
               ('min_pred', Minimum, lambda output: apply_sigmoid(output)[0],{}),
               ('mean_pred', Average, lambda output: apply_sigmoid(output)[0], {}),
               ('max_pred', Maximum, lambda output: apply_sigmoid(output)[0],{})],

    'multi': [
        ('accuracy', Accuracy, binarize_softmax, {}),
        ('ap', Precision, binarize_softmax, {'average': True}),
        ('ar', Recall, binarize_softmax, {'average': True})
    ]

}


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
                   add_loss=True, loss_kwargs_mapping=None):
    if metric_mapping == None:
        metric_mapping = default_metric_mapping

    # should return only one, but specified multiple --> error
    # should return multiple, but specified only one --> no error
    if ~extract_multiple_atts & (len(attribute_specs) > 1):
        raise ValueError(f'Was asked to extract single attribute, but found {len(attribute_specs)}')

    final_metrics = {}
    for att_spec in attribute_specs:
        if loss_kwargs_mapping and att_spec.name in loss_kwargs_mapping:
            loss_kwargs = loss_kwargs_mapping[att_spec.name]
        else:
            loss_kwargs = {}
        if add_loss:
            # make loss metric
            loss_name = 'loss_' + att_spec.name
            loss_fn = loss_mapping[att_spec.target_type]
            loss_fn = loss_fn(**loss_kwargs)
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