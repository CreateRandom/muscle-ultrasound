import operator
import os
import random
from functools import reduce

import numpy as np
import itertools

from numpy import nanmedian, nanstd
from scipy.stats import describe, scoreatpercentile
import torch

from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def repeat_to_length(elems, length):

    n_repeats = int(np.ceil(length / len(elems)))
    repeated = list(itertools.repeat(elems, n_repeats))
    repeated = flatten_list(repeated)
    return repeated[:length]

def flatten_list(elems):
    return reduce(operator.add, elems)


def compute_normalization_parameters(dataset, n_channels):
    full_loader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=os.cpu_count())
    from tqdm import tqdm
    mean = torch.zeros(1)
    std = torch.zeros(1)
    print('==> Computing mean and std..')
    for inputs, _labels in tqdm(full_loader):
        for i in range(n_channels):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def pytorch_count_params(model):
  "count number trainable parameters in a pytorch model"
  total_params = sum(reduce( lambda a, b: a*b, x.size()) for x in model.parameters())
  return total_params

def geometric_mean(input_x, dim):
    log_x = torch.log(input_x)
    return torch.exp(torch.mean(log_x, dim=dim))

def extract_simple_descriptive_features(vector, prefix=None):
    result = describe(vector)
    # get the fields out and transform them into dict entries
    result_dict = dict((name, getattr(result, name)) for name in dir(result) if
                       not name.startswith('_') and not name.startswith('__') and not callable(getattr(result, name)))
    result_dict['min'] = result_dict['minmax'][0]
    result_dict['max'] = result_dict['minmax'][1]
    result_dict.pop('minmax')
    if prefix:
        new_dict = {}
        for k, v in result_dict.items():
            new_dict[prefix + '_' + k] = v
        result_dict = new_dict
    return result_dict

def fivenum(vector):
    """Returns Tukey's five number summary
    (minimum, lower-hinge, median, upper-hinge, maximum) for the input vector,
    a list or array of numbers based on 1.5 times the interquartile distance"""
    try:
        np.sum(vector)
    except TypeError:
        print('Error: you must provide a list or array of only numbers')
    q1 = scoreatpercentile(vector[~np.isnan(vector)], 25)
    q3 = scoreatpercentile(vector[~np.isnan(vector)], 75)
    iqd = q3-q1
    md = nanmedian(vector)
    whisker = 1.5*iqd
    return {'min': np.nanmin(vector), 'lower_hinge': md - whisker, 'median':md, 'upper_hinge': md + whisker, 'max': np.nanmax(vector)}


def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)