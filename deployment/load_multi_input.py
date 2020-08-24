import os

import torch

from baselines import get_default_set_spec_dict, evaluate_roc
from loading.datasets import make_att_specs
from loading.loaders import make_basic_transform, get_data_for_spec, make_bag_dataset, wrap_in_bag_loader
from models.multi_input import MultiInputNet, cnn_constructors, MultiInputBaseline
from utils.binarize_utils import _apply_sigmoid
import pandas as pd

def load_checkpoint(checkpoint_dict_or_path):
    if not isinstance(checkpoint_dict_or_path, dict):
        checkpoint_dict_or_path = torch.load(checkpoint_dict_or_path)
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

def load_transform_from_checkpoint(checkpoint_dict_or_path, ignore_resize=True):
    combined_dict = load_checkpoint(checkpoint_dict_or_path)

    transform_dict = combined_dict['transform_dict']
    # decide whether to override the resize
    if ignore_resize:
        transform_dict['resize_option_name'] = None
    transform = make_basic_transform(**transform_dict)
    return transform

if __name__ == '__main__':

    base_checkpoint_path = '/mnt/chansey/klaus/muscle-ultrasound/checkpoints/'
    project_name = '' #'createrandom/mus-experiments'
    experiment = ''# 'MUS2-25'
    file_name = 'pref_checkpoint'
    epoch = 2

    file_name = file_name + '_' + str(epoch) + '.pt'

    # build path to the checkpoint
    checkpoint_path = os.path.join(base_checkpoint_path,project_name,experiment,file_name)
    checkpoint_dict = load_checkpoint(checkpoint_path)

    print('Loaded checkpoint.')

    config = checkpoint_dict['config']

    prob_type = config['problem_type']
    # find out what type of model we need to load
    if prob_type == 'bag':
        model = load_multi_input(checkpoint_dict)
    elif prob_type == 'image':
        model = load_image_baseline(checkpoint_dict)
    else:
        raise ValueError(f'Unknown problem type {prob_type}')

    transform = load_transform_from_checkpoint(checkpoint_dict,ignore_resize=False)

    model = model.eval()

    set_spec = get_default_set_spec_dict()['Philips_iU22_val']
    patients = get_data_for_spec(set_spec, loader_type='bag')
    patients = patients[0:10]
    all_specs = make_att_specs()
    att_spec = all_specs['Class']
    muscle_spec = all_specs['Muscle']
    side_spec = all_specs['Side']
    ds = make_bag_dataset(patients, set_spec.img_root_path, use_one_channel=(config['in_channels'] == 1),
                          attribute_specs=[att_spec, muscle_spec, side_spec], transform=transform,
                          use_pseudopatients=False,
                          return_attribute_dict=True)

    loader = wrap_in_bag_loader(ds, 1, pin_memory=True, return_attribute_dict=True, shuffle=False)

    attention_ei_frames = []
    image_ei_frames = []
    y_pred = []
    y_true = []
    for i, (imgs,y) in enumerate(loader):
        print(i)
      #  n_img = torch.tensor([imgs.shape[0]])
      #  pred_output = model((imgs, n_img))
        pred_output = model(imgs)
        # get the main prediction
        if isinstance(pred_output,dict):
            if prob_type == 'bag':
                class_pred = pred_output['head_preds']['Class']
            elif prob_type == 'image':
                class_pred = pred_output['preds']
            else:
                raise ValueError()
        else:
            class_pred = pred_output
        p = _apply_sigmoid(class_pred)
        y_pred.extend(p.detach().numpy().tolist())
        y_true.extend(y['Class'].detach().numpy().tolist())

        muscles = y['Muscle']
        sides = y['Side']
        ei_frame = patients[i].get_selected_record().get_EI_frame()
        # get sub-level predictions
        if 'attention_outputs' in pred_output:
            att_scores = pred_output['attention_outputs'][0].numpy().tolist()[0]
            att_frame = pd.DataFrame({'Side': sides, 'Muscle': muscles,'Attention': att_scores})

            total = pd.merge(att_frame, ei_frame, on=['Muscle', 'Side'])
            attention_ei_frames.append(total)

        if 'image_preds' in pred_output:
            image_scores = pred_output['image_preds'][0].flatten().numpy().tolist()
            im_frame = pd.DataFrame({'Side': sides, 'Muscle': muscles, 'ImagePrediction': image_scores})
            total = pd.merge(im_frame, ei_frame, on=['Muscle', 'Side'])
            image_ei_frames.append(total)

    if attention_ei_frames:
        total_att_ei = pd.concat(attention_ei_frames)

    if image_ei_frames:
        total_image_ei = pd.concat(image_ei_frames)
    print(y_pred)

    # export the ground truth and the predictions

    os.makedirs('roc_analysis/preds', exist_ok=True)

    pd.DataFrame(y_true,columns=['true']).to_csv('roc_analysis/y_true.csv', index=False, header=True)
    pred_path = os.path.join('roc_analysis','preds',(experiment + '.csv'))
    pd.DataFrame(y_pred, columns=['pred']).to_csv(pred_path, index=False, header=True)

    evaluate_roc(y_true, y_pred, experiment)


