import torch

from loading.loaders import make_basic_transform
from models.multi_input import MultiInputNet, cnn_constructors, MultiInputBaseline


def load_checkpoint(checkpoint_dict_or_path,device=None):
    if not isinstance(checkpoint_dict_or_path, dict):
        checkpoint_dict_or_path = torch.load(checkpoint_dict_or_path, map_location=device)
    return checkpoint_dict_or_path

def load_image_baseline(checkpoint_dict_or_path):
    """
        For loading an instance-level aggregation model obtained via train_image_level.py
    """
    combined_dict = load_checkpoint(checkpoint_dict_or_path)
    config = combined_dict['config']
    backend = config['backend']
    backend_func = cnn_constructors[backend]
    model_param_dict = combined_dict['model_param_dict']
    model = backend_func(**model_param_dict)
    state_dict = combined_dict['model_state_dict']
    model.load_state_dict(state_dict)
    # wrap in the multi-input baseline
    model = MultiInputBaseline(model, label_type=config['label_type'])
    return model

def load_multi_input(checkpoint_dict_or_path):
    combined_dict = load_checkpoint(checkpoint_dict_or_path)
    model_param_dict = combined_dict['model_param_dict']
    model = MultiInputNet(**model_param_dict)
    state_dict = combined_dict['model_state_dict']
    model.load_state_dict(state_dict)
    return model


def load_transform_dict(checkpoint_dict_or_path, resize_option_name=None, ignore_resize=True):
    combined_dict = load_checkpoint(checkpoint_dict_or_path)

    transform_dict = combined_dict['transform_dict']

    # reset if so desired
    if resize_option_name:
        transform_dict['resize_option_name'] = resize_option_name
    # decide whether to override the resize
    if ignore_resize:
        transform_dict['resize_option_name'] = None
    return transform_dict


def load_transform_from_checkpoint(checkpoint_dict_or_path, resize_option_name=None, ignore_resize=True):
    transform_dict = load_transform_dict(checkpoint_dict_or_path, resize_option_name=resize_option_name, ignore_resize=ignore_resize)
    transform = make_basic_transform(**transform_dict)
    return transform