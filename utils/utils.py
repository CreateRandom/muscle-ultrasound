import operator
import os
from functools import reduce

import numpy as np
import itertools

import torch
from tqdm import tqdm


def repeat_to_length(elems, length):

    n_repeats = int(np.ceil(length / len(elems)))
    repeated = list(itertools.repeat(elems, n_repeats))
    repeated = flatten_list(repeated)
    return repeated[:length]

def flatten_list(elems):
    return reduce(operator.add, elems)


def compute_normalization_parameters(dataset, n_channels):
    full_loader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=os.cpu_count())

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