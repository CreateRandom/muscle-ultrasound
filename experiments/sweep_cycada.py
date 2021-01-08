import random

import ray
from ray import tune
from ray.tune import sample_from
from ray.ray_constants import OBJECT_STORE_MINIMUM_MEMORY_BYTES

from experiments.experiment_defaults import esaote_train, philips_train
from train_multi_input import train_multi_input

def sweep_cycada(num_samples):
    # see here https://github.com/ray-project/ray/issues/7084
    ray.init(webui_host='127.0.0.1', object_store_memory=OBJECT_STORE_MINIMUM_MEMORY_BYTES)

    train_set_specs = [esaote_train, philips_train]

    # NB: fill in paths of the mapped images
    mapped_paths = {'mapped_source_path': 'some_path', 'mapped_target_path': 'some_other_path'}
    # iterate over mapping the source or the target
    for to_map, mapped_path in mapped_paths.items():
        for train_set_spec in train_set_specs:
            # run single head classification with no adjustments

            base_config = {'prediction_target': 'Class', 'backend_mode': 'scratch',
                           'backend': 'resnet-18', 'n_epochs': 15, 'neptune_project': 'createrandom/mus-cycada',
                           'problem_type': 'bag', 'batch_size': 8, 'mil_mode': 'embedding',
                           'use_pseudopatients': True, 'fc_use_bn': True, 'fc_hidden_layers': 2,
                           'backend_cutoff': 1, 'mil_pooling': 'attention', to_map: mapped_path}

            base_config = {**base_config, **train_set_spec}



            bag_sweep_config = {
                # lr for bag classifier and pooling
                "lr": sample_from(lambda x: random.uniform(0.001, 0.1)),
                # effective extract_only should be possible by setting a very small lr
                "backend_lr": sample_from(lambda x: random.uniform(0.001, 0.1)),
                "attention_mode": sample_from(lambda x: random.choice(['sigmoid', 'softmax'])),
            }

            config = {**base_config, **bag_sweep_config}

            tune.run(train_multi_input,
                     config=config,
                     num_samples=num_samples,
                     resources_per_trial={"gpu": 1, "cpu": 8})

    ray.shutdown()
