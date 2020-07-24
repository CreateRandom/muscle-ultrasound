import os

from deployment.load_multi_input import load_checkpoint
from train_image_level import train_image_level
from train_multi_input import train_multi_input

if __name__ == '__main__':
    base_path = '/mnt/chansey/klaus/muscle-ultrasound/checkpoints/'

    project_name = 'createrandom/MUS-RQ1'

    experiment = 'MUS1-518'

    file_name = 'pref_checkpoint'
    epoch = 10

    file_name = file_name + '_' + str(epoch) + '.pt'

    # build path to the checkpoint
    checkpoint_root = os.path.join(base_path, project_name, experiment)
    checkpoint_path = os.path.join(checkpoint_root,file_name)
    new_checkpoint_path = os.path.join(checkpoint_root, 'retrain_val')
    # load the config from the old checkpoint
    checkpoint_dict = load_checkpoint(checkpoint_path)

    config = checkpoint_dict['config']

    # include the val set now
    config['source_train'] = config['source_train'] + '+val'
    # add the new checkpoint path
    config['checkpoint_dir'] = new_checkpoint_path
    if config['problem_type'] == 'image':
        train_image_level(config)
    else:
        train_multi_input(config)