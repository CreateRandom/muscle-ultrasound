import random

import ray
from ray import tune
from ray.tune import sample_from
from ray.ray_constants import OBJECT_STORE_MINIMUM_MEMORY_BYTES

from train_image_level import train_image_level
from train_multi_input import train_multi_input

def run_rq1():
    # see here https://github.com/ray-project/ray/issues/7084
    ray.init(webui_host='127.0.0.1', object_store_memory=OBJECT_STORE_MINIMUM_MEMORY_BYTES)

    num_samples = 12

    attributes = ['Class']

    for attribute in attributes:

        base_config = {'prediction_target': attribute, 'backend_mode': 'finetune',
                  'backend': 'resnet-18', 'n_epochs': 10, 'neptune_project': 'createrandom/MUS-RQ1'}


        esaote_train = {'source_train': 'ESAOTE_6100_train',
                             'val': ['ESAOTE_6100_val', 'Philips_iU22_val']}

        philips_train = {'source_train': 'Philips_iU22_train',
                             'val': ['ESAOTE_6100_val', 'Philips_iU22_val']}

        train_set_spec = esaote_train

        base_config = {**base_config, **train_set_spec}


        image_config = {'problem_type': 'image', 'batch_size': 32}

        image_config = {**base_config, **image_config}

        image_sweep_config = {"lr": sample_from(lambda x: random.uniform(0.001, 0.1))}

        config = {**image_config, **image_sweep_config}


        tune.run(train_image_level,
                 config=config,
                 num_samples=num_samples,
                 resources_per_trial={"gpu": 1, "cpu": 8})

        # decide what to run here

        bag_config_mean = {'problem_type': 'bag', 'batch_size': 8, 'mil_mode': 'embedding',
                      'use_pseudopatients': True, 'fc_use_bn': True, 'fc_hidden_layers': 2,
                      'backend_cutoff': 1}

        bag_config_mean = {**base_config, **bag_config_mean, **{'mil_pooling': 'mean'}}

        bag_config_att = {**base_config, **bag_config_mean, **{'mil_pooling': 'attention', 'attention_mode': 'sigmoid'}}

        bag_sweep_config = {
            # lr for bag classifier and pooling
            "lr": sample_from(lambda x: random.uniform(0.001, 0.1)),
            # effective extract_only should be possible by setting a very small lr
            "backend_lr": sample_from(lambda x: random.uniform(0.001, 0.1))
        }

        config = {**bag_config_mean, **bag_sweep_config}

        tune.run(train_multi_input,
                 config=config,
                 num_samples=num_samples,
                 resources_per_trial={"gpu": 1, "cpu": 8})

        config = {**bag_config_att, **bag_sweep_config}

        tune.run(train_multi_input,
                 config=config,
                 num_samples=num_samples,
                 resources_per_trial={"gpu": 1, "cpu": 8})
