from torch import nn

import torch


def _binarize_softmax(y_pred):
    sm = nn.Softmax(dim=1)(y_pred)
    _, i = torch.max(sm, dim=1)
    return i

def _apply_sigmoid(y_pred):
    y_pred = nn.Sigmoid()(y_pred)
    return y_pred

def _binarize_sigmoid(y_pred):
    y_pred = _apply_sigmoid(y_pred)
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

def apply_sigmoid(output):
    y_pred, y = output['y_pred'], output['y']
    y_pred = _apply_sigmoid(y_pred)
    return (y_pred, y)