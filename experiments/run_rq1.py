import random

import ray
from ray import tune
from ray.tune import sample_from
from copy import deepcopy

from train_multi_input import train_model

if __name__ == '__main__':
    # see here https://github.com/ray-project/ray/issues/7084
    ray.init(webui_host='127.0.0.1')

    base_config = {'prediction_target': 'Sex', 'backend_mode': 'finetune',
              'backend': 'resnet-18', 'n_epochs': 10, 'neptune_project': 'createrandom/MUS-RQ1'}

    image_config = {'problem_type': 'image', 'batch_size': 64}

    image_config = {**base_config, **image_config}

    image_sweep_config = {"backend_lr": sample_from(lambda x: random.uniform(0.001, 0.1))}

    # decide what to run here

    bag_config = {'problem_type': 'bag', 'batch_size': 8, 'mil_mode': 'embedding', 'mil_pooling': 'mean',
                  'use_pseudopatients': True, 'fc_use_bn': True, 'fc_hidden_layers': 2,
                  'backend_cutoff': 1}

    bag_sweep_config = {
        # lr for bag classifier and pooling
        "lr": sample_from(lambda x: random.uniform(0.001, 0.1)),
        # effective extract_only should be possible by setting a very small lr
        "backend_lr": sample_from(lambda x: random.uniform(0.001, 0.1))
    }

    config = {**image_config, **image_sweep_config}

    num_samples = 24
    tune.run(train_model,
                        config=config,
                        num_samples=num_samples,
                        resources_per_trial={"gpu": 1, "cpu": 8})

    config = {**bag_config, **bag_sweep_config}

    tune.run(train_model,
                        config=config,
                        num_samples=num_samples,
                        resources_per_trial={"gpu": 1, "cpu": 8})