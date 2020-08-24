import random

import ray
from ray import tune
from ray.tune import sample_from
from ray.ray_constants import OBJECT_STORE_MINIMUM_MEMORY_BYTES

from experiments.experiment_defaults import esaote_train, philips_train
from train_image_level import train_image_level

def sweep_baseline(num_samples):
    # see here https://github.com/ray-project/ray/issues/7084
    ray.init(webui_host='127.0.0.1', object_store_memory=OBJECT_STORE_MINIMUM_MEMORY_BYTES)

    train_set_specs = [esaote_train, philips_train]

    base_config = {'prediction_target': 'Class', 'backend': 'resnet-18', 'n_epochs': 15,
                   'neptune_project': 'createrandom/mus-imageagg', 'batch_size': 32}

    for train_set_spec in train_set_specs:

        total_config = {**base_config, **train_set_spec}

        image_sweep_config = {"lr": sample_from(lambda x: random.uniform(0.001, 0.1)),
                              "backend_mode": sample_from(lambda x: random.choice(['finetune', 'scratch']))}

        config = {**total_config, **image_sweep_config}

        tune.run(train_image_level,
                 config=config,
                 num_samples=num_samples,
                 resources_per_trial={"gpu": 1, "cpu": 8})

    ray.shutdown()