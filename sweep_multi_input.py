import random

import ray
from ray import tune
from ray.tune import sample_from

from train_multi_input import train_model

from torchvision.datasets import ImageNet
if __name__ == '__main__':
    # see here https://github.com/ray-project/ray/issues/7084
    ray.init(webui_host='127.0.0.1')

    # only top-level, fine-tune pre-trained, train from scratch

    # sweep over lr, backend

    base_config = {'problem_type': 'bag','prediction_target': 'Sex', 'backend_mode': 'extract_only',
              'backend': 'resnet-18', 'mil_pooling': 'mean',
              'mil_mode': 'embedding', 'batch_size': 4, 'lr': 0.1, 'n_epochs': 5,
              'use_pseudopatients': False, 'data_source': 'umc', 'fc_use_bn': True, 'fc_hidden_layers': True}


    sweep_config = {
        "lr": sample_from(lambda x: random.uniform(0.001, 0.1)),
        "backend_mode": sample_from(lambda x: random.choice(['finetune', 'extract_only']))
    }

    config = {**base_config, **sweep_config}

    num_samples = 10
    analysis = tune.run(train_model,
                        config=config,
                        num_samples=num_samples,
                        resources_per_trial={"gpu": 1, "cpu": 8})