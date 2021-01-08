import os

from inference.conditions import trained_on_esaote
from utils.experiment_utils import get_mnt_path
from inference.load_multi_input import load_checkpoint
from train_image_level import train_image_level
from train_multi_input import train_multi_input

if __name__ == '__main__':

    mnt_path = get_mnt_path()
    checkpoint_folder = 'klaus/muscle-ultrasound/checkpoints/'
    neptune_user = 'createrandom'
    base_path = os.path.join(mnt_path,checkpoint_folder)

    check_prefix = 'pref_checkpoint'

    c = trained_on_esaote[0]
    file_name = check_prefix + '_' + str(c.epoch) + '.pt'
    # build path to the checkpoint
    checkpoint_root = os.path.join(mnt_path, checkpoint_folder, neptune_user, c.project_name, c.condition_name)
    checkpoint_path = os.path.join(checkpoint_root,file_name)
    new_checkpoint_path = os.path.join(checkpoint_root, 'retrain_val')
    # load the config from the old checkpoint
    checkpoint_dict = load_checkpoint(checkpoint_path)

    config = checkpoint_dict['config']
    mode = 'evaluate_mask'
    if mode == 'retrain':
        # include the val set now
        config['source_train'] = config['source_train'] + '+val'
        # add the new checkpoint path
        config['checkpoint_dir'] = new_checkpoint_path
    elif mode == 'evaluate_coral':
        target_train = 'Philips_iU22_train' if config['source_train'] == 'ESAOTE_6100_train' else 'ESAOTE_6100_train'
        config['target_train'] = target_train
        config['lambda_weight'] = 2.75
        config['layers_to_compute_da_on'] = [2]
        config['neptune_project'] = 'createrandom/mus-coral'
    elif mode == 'evaluate_mask':
        config['use_mask'] = True
        config['neptune_project'] = 'createrandom/mus-simplemil'

    if config['problem_type'] == 'image':
        train_image_level(config)
    else:
        train_multi_input(config)