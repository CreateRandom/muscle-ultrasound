import os

import pandas as pd
from sklearn.metrics import roc_auc_score

from baselines.domain_mapping import get_brightness_factor, get_lr_model, get_mapped_path
from baselines.evaluation import evaluate_roc, find_threshold_for_specificity

from inference.conditions import conditions_to_run
from inference.load_multi_input import load_checkpoint, load_multi_input, load_image_baseline, load_transform_dict, \
    load_transform_from_checkpoint
from loading.datasets import make_att_specs
from loading.loaders import make_basic_transform, get_data_for_spec, make_bag_dataset, wrap_in_bag_loader
from utils.binarize_utils import _apply_sigmoid
from utils.experiment_utils import get_default_set_spec_dict, get_mnt_path


def load_transform_with_brightness_adjustment(transform_dict, source, target):
    brightness_factor = get_brightness_factor(source, target)
    transform_dict['brightness_factor'] = brightness_factor
    transform = make_basic_transform(**transform_dict)
    return transform


def load_transform_with_regression_adjustment(transform_dict, source, target):
    lr_model = get_lr_model(source, target)
    transform_dict['regression_model'] = lr_model
    transform = make_basic_transform(**transform_dict)
    return transform

def test_checkpoint(condition, checkpoint_dict, adjustment, source, target, set_name, test_run=False):
    '''
    Tests an experimental condition on the specified set, using the specified adjustment.
    :param condition: the experimental run to test
    :param checkpoint_dict: A checkpoint dictionary
    :param adjustment: an adjustment method
    :param source: the source set
    :param target: the target set
    :param set_name: what slice to use, e.g. test or val
    :param test_run: if true, use a small subset
    :return: None
    '''
    config = checkpoint_dict['config']

    prob_type = config['problem_type']
    # find out what type of model we need to load
    if prob_type == 'bag':
        model = load_multi_input(checkpoint_dict)
    elif prob_type == 'image':
        model = load_image_baseline(checkpoint_dict)
    else:
        raise ValueError(f'Unknown problem type {prob_type}')

    eval_set = target + '_' + set_name

    set_spec_dict = get_default_set_spec_dict()
    set_spec = set_spec_dict[eval_set]

    if adjustment == 'brightness':
        transform_dict = load_transform_dict(checkpoint_dict, resize_option_name=set_spec.device,
                                             ignore_resize=False)

        transform = load_transform_with_brightness_adjustment(transform_dict, source=source,
                                                              target=target)
    elif adjustment == 'regression':
        transform_dict = load_transform_dict(checkpoint_dict, resize_option_name=set_spec.device,
                                             ignore_resize=False)
        transform = load_transform_with_regression_adjustment(transform_dict, source=source,
                                                              target=target)
    # standard transform
    else:
        transform = load_transform_from_checkpoint(checkpoint_dict, resize_option_name=set_spec.device,
                                                   ignore_resize=False)

    if adjustment == 'mapped_images':
        img_root_path = get_mapped_path(source, target)
        enforce_all_images_exist = False
        strip_folder = True
    else:
        img_root_path = set_spec.img_root_path
        enforce_all_images_exist = True
        strip_folder = False

    model = model.eval()

    patients = get_data_for_spec(set_spec, loader_type='bag')
    if test_run:
        patients = patients[0:10]
    # patients = patients[0:10]
    all_specs = make_att_specs()
    att_spec = all_specs['Class']
    muscle_spec = all_specs['Muscle']
    side_spec = all_specs['Side']
    ds = make_bag_dataset(patients, img_root_path, use_one_channel=(config['in_channels'] == 1),
                          attribute_specs=[att_spec, muscle_spec, side_spec], transform=transform,
                          use_pseudopatients=False,
                          return_attribute_dict=True, strip_folder=strip_folder,
                          enforce_all_images_exist=enforce_all_images_exist)

    loader = wrap_in_bag_loader(ds, 1, pin_memory=True, return_attribute_dict=True, shuffle=False)

    attention_ei_frames = []
    image_ei_frames = []
    y_pred = []
    y_true = []
    for i, (imgs, y) in enumerate(loader):
        print(i)
        #  n_img = torch.tensor([imgs.shape[0]])
        #  pred_output = model((imgs, n_img))
        pred_output = model(imgs)
        # get the main prediction
        if isinstance(pred_output, dict):
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
        record = patients[i].get_selected_record()
        ei_frame = record.get_EI_frame()
        if 'pid' in record.meta_info:
            pid = record.meta_info['pid']
        else:
            pid = i
        if not ei_frame.empty:
            # get sub-level predictions
            if 'attention_outputs' in pred_output:
                att_scores = pred_output['attention_outputs'][0].numpy().tolist()[0]
                att_frame = pd.DataFrame({'Side': sides, 'Muscle': muscles, 'Score': att_scores, 'pid': pid})

                total = pd.merge(att_frame, ei_frame, on=['Muscle', 'Side'])
                attention_ei_frames.append(total)

            if 'image_preds' in pred_output:
                image_scores = pred_output['image_preds'][0].flatten().numpy().tolist()
                im_frame = pd.DataFrame({'Side': sides, 'Muscle': muscles, 'Score': image_scores, 'pid': pid})
                total = pd.merge(im_frame, ei_frame, on=['Muscle', 'Side'])
                image_ei_frames.append(total)
    score_frame = None
    if attention_ei_frames:
        score_frame = pd.concat(attention_ei_frames)

    if image_ei_frames:
        score_frame = pd.concat(image_ei_frames)
    print(y_pred)

    # export the ground truth and the predictions
    export_path = os.path.join('../roc_analysis', eval_set)
    os.makedirs(export_path, exist_ok=True)

    pd.DataFrame(y_true, columns=['true']).to_csv(os.path.join(export_path, 'y_true.csv'), index=False, header=True)
    pred_path = os.path.join(export_path, 'proba')
    # add a separate folder for in-domain vs out-of-domain
    extra_folder = 'in_domain' if source == target else 'out_domain'
    pred_path = os.path.join(pred_path, extra_folder)
    os.makedirs(pred_path, exist_ok=True)
    exp_name = condition.condition_name.replace("/", "_") + '_' + str(condition.epoch)
    adjustment_name = adjustment if adjustment else ''
    csv_path = os.path.join(pred_path, (exp_name + '_' + adjustment_name + '.csv'))
    pd.DataFrame(y_pred, columns=['pred']).to_csv(csv_path, index=False, header=True)

    if score_frame is not None:
        score_path = os.path.join(pred_path, (exp_name + '_' + adjustment_name + 'scores.csv'))
        score_frame.to_csv(score_path)

    find_threshold_for_specificity(y_true,y_pred,0.8)
    evaluate_roc(y_true, y_pred, condition.condition_name)

    print(roc_auc_score(y_true, y_pred))

def load_checkpoint_for_condition(condition):
    mnt_path = get_mnt_path()
    checkpoint_folder = 'klaus/muscle-ultrasound/checkpoints/'
    neptune_user = 'createrandom'

    check_prefix = 'pref_checkpoint'

    file_name = check_prefix + '_' + str(condition.epoch) + '.pt'
    # build path to the checkpoint
    checkpoint_root = os.path.join(mnt_path, checkpoint_folder, neptune_user, condition.project_name,
                                   condition.condition_name)
    checkpoint_path = os.path.join(checkpoint_root, file_name)

    checkpoint_dict = load_checkpoint(checkpoint_path)

    print('Loaded checkpoint.')

    return checkpoint_dict

def main():

    set_name = 'test'  # can specify 'test' or 'val'

    # adjustment methods to use for Esaote and Philips, empty string is no adjustment
    adjustments_e_and_p = ['', 'brightness', 'regression']#, 'mapped_images']
    # adjustment methods to use for GE
    adjustments_g = ['', 'brightness']

    for source_domain, conditions in conditions_to_run.items():
        target_domain = 'Philips_iU22' if source_domain == 'ESAOTE_6100' else 'ESAOTE_6100'
        print(f'Running on source domain: {source_domain}')
        for condition in conditions:
            checkpoint_dict = load_checkpoint_for_condition(condition)
            print(f'Running condition: {condition.condition_name}-{condition.epoch}')
            # evaluate in-domain performance
            test_checkpoint(condition, checkpoint_dict, adjustment='', source=source_domain, target=source_domain, set_name=set_name)
            # try various adjustments
            for adjustment in adjustments_e_and_p:
                print(f'Running adjustment method on {target_domain}: {adjustment}')
                test_checkpoint(condition, checkpoint_dict, adjustment=adjustment, source=source_domain, target=target_domain, set_name=set_name)

            # evaluate GE Loqiq set
            for adjustment in adjustments_g:
                print(f'Running adjustment method on GE: {adjustment}')
                test_checkpoint(condition, checkpoint_dict, adjustment=adjustment, source=source_domain, target='GE_Logiq_E', set_name='im_muscle_chart')

if __name__ == '__main__':
    main()
