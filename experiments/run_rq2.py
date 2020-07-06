import random

import ray
from ray import tune
from ray.tune import sample_from
from ray.ray_constants import OBJECT_STORE_MINIMUM_MEMORY_BYTES

from train_multi_input import train_multi_input
from utils.utils import powerset

attributes_to_include = ['Age', 'Sex', 'BMI']
loss_weight_ranges = {'Sex' : (0.1, 3), 'Age': (0.0001, 1), 'BMI': (0.0001,1)}

def sample_loss_weight_dict():
    combs = list(powerset(attributes_to_include))
    # remove the empty set
    combs.remove(())

    comb = random.choice(combs)
    comb = list(comb)
    loss_weight_dict = {}
    for elem in comb:
        range_interval = loss_weight_ranges[elem]
        weight = random.uniform(*range_interval)
        loss_weight_dict[elem] = weight
    return loss_weight_dict

def run_rq1():
    # see here https://github.com/ray-project/ray/issues/7084
    ray.init(webui_host='127.0.0.1', object_store_memory=OBJECT_STORE_MINIMUM_MEMORY_BYTES)

    num_samples = 12

    attribute = 'Class'

    esaote_train = {'source_train': 'ESAOTE_6100_train',
                    'val': ['ESAOTE_6100_val', 'Philips_iU22_val']}

    philips_train = {'source_train': 'Philips_iU22_train',
                     'val': ['ESAOTE_6100_val', 'Philips_iU22_val']}

    train_set_specs = [esaote_train]#, philips_train]

    for train_set_spec in train_set_specs:
        base_config = {'prediction_target': attribute, 'backend_mode': 'finetune',
                       'backend': 'resnet-18', 'n_epochs': 10, 'neptune_project': 'createrandom/MUS-RQ1'}

        base_config = {**base_config, **train_set_spec}



        bag_config = {'problem_type': 'bag', 'batch_size': 8, 'mil_mode': 'embedding',
                           'use_pseudopatients': True, 'fc_use_bn': True, 'fc_hidden_layers': 2,
                           'backend_cutoff': 1, 'mil_pooling': 'attention', 'attention_mode': 'sigmoid'}

        bag_config = {**base_config, **bag_config}

        # run multi head classification, varying which classes are included
        bag_sweep_config = {
            # lr for bag classifier and pooling
            "lr": sample_from(lambda x: random.uniform(0.001, 0.1)),
            # effective extract_only should be possible by setting a very small lr
            "backend_lr": sample_from(lambda x: random.uniform(0.001, 0.1)),
            "att_loss_weights": sample_from(lambda x: sample_loss_weight_dict())
        }

        config = {**bag_config, **bag_sweep_config}

        tune.run(train_multi_input,
                 config=config,
                 num_samples=num_samples,
                 resources_per_trial={"gpu": 1, "cpu": 8})