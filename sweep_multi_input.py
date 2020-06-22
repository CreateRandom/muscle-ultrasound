import random

import ray
from ray import tune
from ray.tune import sample_from

from train_multi_input import train_model

if __name__ == '__main__':
    # see here https://github.com/ray-project/ray/issues/7084
    ray.init(webui_host='127.0.0.1')

    # only top-level, fine-tune pre-trained, train from scratch

    # sweep over lr, backend

    base_config = {'problem_type': 'bag','prediction_target': 'Sex', 'backend_mode': 'finetune',
              'backend': 'resnet-18', 'mil_pooling': 'attention',
              'mil_mode': 'embedding', 'batch_size': 8, 'lr': 0.1, 'n_epochs': 5,
              'use_pseudopatients': True, 'data_source': 'umc', 'fc_use_bn': True, 'fc_hidden_layers': 2,
              'backend_cutoff': 1}


    sweep_config = {
        # lr for bag classifier and pooling
        "lr": sample_from(lambda x: random.uniform(0.001, 0.1)),
        # effective extract_only should be possible by setting a very small lr
        "backend_lr": sample_from(lambda x: random.uniform(0.001, 0.1)),
        "attention_mode": sample_from(lambda x: random.choice(['sigmoid', 'identity', 'softmax'])),
    }

    config = {**base_config, **sweep_config}

    num_samples = 24
    analysis = tune.run(train_model,
                        config=config,
                        num_samples=num_samples,
                        resources_per_trial={"gpu": 1, "cpu": 8})