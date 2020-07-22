import torch

from baselines import get_default_set_spec_dict, evaluate_preds
from loading.datasets import make_att_specs
from loading.loaders import make_basic_transform, get_data_for_spec, make_bag_dataset, wrap_in_bag_loader
from models.multi_input import MultiInputNet, cnn_constructors, MultiInputBaseline
from utils.binarize_utils import _apply_sigmoid


def load_image_baseline(checkpoint_path):
    """
        For loading an instance-level aggregation model obtained via train_image_level.py
    """
    combined_dict = torch.load(checkpoint_path)
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

def load_multi_input(checkpoint_path):
    combined_dict = torch.load(checkpoint_path)

    model_param_dict = combined_dict['model_param_dict']
    model = MultiInputNet(**model_param_dict)
    state_dict = combined_dict['model_state_dict']
    model.load_state_dict(state_dict)
    return model

def load_transform_from_checkpoint(checkpoint_path, ignore_resize=True):
    combined_dict = torch.load(checkpoint_path)
    transform_dict = combined_dict['transform_dict']
    # decide whether to override the resize
    if ignore_resize:
        transform_dict['resize_option_name'] = None
    transform = make_basic_transform(**transform_dict)
    return transform

def load_config_from_checkpoint(checkpoint_path):
    combined_dict = torch.load(checkpoint_path)
    config = combined_dict['config']
    return config

if __name__ == '__main__':
    checkpoint_path = '../checkpoints/pref_checkpoint_4.pt'
    config = load_config_from_checkpoint(checkpoint_path)

    prob_type = config['problem_type']
    # find out what type of model we need to load
    if prob_type == 'bag':
        model = load_multi_input(checkpoint_path)
    elif prob_type == 'image':
        model = load_image_baseline(checkpoint_path)
    else:
        raise ValueError(f'Unknown problem type {prob_type}')

    transform = load_transform_from_checkpoint(checkpoint_path,ignore_resize=False)

    model = model.eval()

    set_spec = get_default_set_spec_dict()['Philips_iU22_val']
    patients = get_data_for_spec(set_spec, loader_type='bag')

    att_spec = make_att_specs()['Class']

    ds = make_bag_dataset(patients, set_spec.img_root_path, use_one_channel=(config['in_channels'] == 1),
                          attribute_specs=[att_spec], transform=transform,
                          use_pseudopatients=False,
                          return_attribute_dict=True)

    loader = wrap_in_bag_loader(ds, 1, pin_memory=True, return_attribute_dict=True)

    imgs, y = ds[0]
    y_pred = []
    y_true = []
    for i, (imgs,y) in enumerate(loader):
        print(i)
      #  n_img = torch.tensor([imgs.shape[0]])
      #  pred_output = model((imgs, n_img))
        pred_output = model(imgs)
        if isinstance(pred_output,dict):
            class_pred = pred_output['head_preds']['Class']
        else:
            class_pred = pred_output
        p = _apply_sigmoid(class_pred)
        y_pred.extend(p.detach().numpy().tolist())
        y_true.extend(y['Class'].detach().numpy().tolist())

    print(y_pred)

    evaluate_preds(y_true, y_pred, 'tester')


