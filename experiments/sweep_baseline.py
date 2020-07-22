import random

import ray
from ray import tune
from ray.tune import sample_from
from ray.ray_constants import OBJECT_STORE_MINIMUM_MEMORY_BYTES

from train_image_level import train_image_level

def sweep_baseline():
    # see here https://github.com/ray-project/ray/issues/7084
    ray.init(webui_host='127.0.0.1', object_store_memory=OBJECT_STORE_MINIMUM_MEMORY_BYTES)

    num_samples = 20

    esaote_train = {'source_train': 'ESAOTE_6100_train',
                    'val': ['ESAOTE_6100_val', 'Philips_iU22_val']}

    philips_train = {'source_train': 'Philips_iU22_train',
                     'val': ['ESAOTE_6100_val', 'Philips_iU22_val']}

    train_set_specs = [esaote_train, philips_train]

    base_config = {'prediction_target': 'Class', 'backend': 'resnet-18', 'n_epochs': 10,
                   'neptune_project': 'createrandom/MUS-RQ1', 'batch_size': 32}

    for train_set_spec in train_set_specs:

        total_config = {**base_config, **train_set_spec}

        image_sweep_config = {"lr": sample_from(lambda x: random.uniform(0.001, 0.1)),
                              "mode": sample_from(lambda x: random.choice(['finetune', 'scratch']))}

        config = {**total_config, **image_sweep_config}

        tune.run(train_image_level,
                 config=config,
                 num_samples=num_samples,
                 resources_per_trial={"gpu": 1, "cpu": 8})