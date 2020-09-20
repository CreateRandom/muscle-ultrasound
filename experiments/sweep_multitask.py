import random

import ray
from ray import tune
from ray.tune import sample_from
from ray.ray_constants import OBJECT_STORE_MINIMUM_MEMORY_BYTES

from experiments.experiment_defaults import esaote_train, philips_train
from train_multi_input import train_multi_input
from utils.utils import powerset

def sample_loss_weight_dict(loss_weight_ranges, sweep_powerset):

    attributes_to_include = list(loss_weight_ranges.keys())
    if sweep_powerset:
        combs = list(powerset(attributes_to_include))
        # remove the empty set
        combs.remove(())
    else:
        combs = [attributes_to_include]

    comb = random.choice(combs)
    comb = list(comb)
    loss_weight_dict = {}
    for elem in comb:
        range_interval = loss_weight_ranges[elem]
        # sample if a range is provided
        if isinstance(range_interval,tuple):
            weight = random.uniform(*range_interval)
        # else, just assume that we use the element directly
        else:
            weight = range_interval
        loss_weight_dict[elem] = weight
    return loss_weight_dict

def sweep_multitask(num_samples, classification_task):
    # see here https://github.com/ray-project/ray/issues/7084
    ray.init(webui_host='127.0.0.1', object_store_memory=OBJECT_STORE_MINIMUM_MEMORY_BYTES)
    if not classification_task:
        # use the original attributes here
        loss_weight_ranges = {'Sex': 1, 'Age': (1, 100), 'BMI': (1, 100)}
    else:
        # use the binned attributes here
        loss_weight_ranges = {'Sex': 1, 'Age_binned': 1, 'BMI_binned': 1}

    train_set_specs = [esaote_train, philips_train]

    for train_set_spec in train_set_specs:
        base_config = {'prediction_target': 'Class', 'backend_mode': 'finetune',
                       'backend': 'resnet-18', 'n_epochs': 15, 'neptune_project': 'createrandom/mus-multitask',
                       'problem_type': 'bag', 'batch_size': 8, 'mil_mode': 'embedding',
                       'use_pseudopatients': True, 'fc_use_bn': True, 'fc_hidden_layers': 2,
                       'backend_cutoff': 1, 'mil_pooling': 'attention', 'attention_mode': 'softmax', 'use_gmean': classification_task}

        base_config = {**base_config, **train_set_spec}

        # run multi head classification, varying which classes are included
        bag_sweep_config = {
            # lr for bag classifier and pooling
            "lr": sample_from(lambda x: random.uniform(0.001, 0.1)),
            # effective extract_only should be possible by setting a very small lr
            "backend_lr": sample_from(lambda x: random.uniform(0.001, 0.1)),
            "att_loss_weights": sample_from(lambda x: sample_loss_weight_dict(loss_weight_ranges, sweep_powerset=False))
        }

        config = {**base_config, **bag_sweep_config}

        tune.run(train_multi_input,
                 config=config,
                 num_samples=num_samples,
                 resources_per_trial={"gpu": 1, "cpu": 8})

    ray.shutdown()