import ray
from ray import tune

from train_multi_input import train_model


if __name__ == '__main__':
    # see here https://github.com/ray-project/ray/issues/7084
    ray.init(webui_host='127.0.0.1')

    config = {'pretrained': True, 'feature_extraction': True, 'use_one_channel': False,
              'backend': 'resnet-18', 'mil_pooling': 'attention', 'classifier': 'fc',
              'mil_mode': 'embedding', 'batch_size': 4, 'lr': 0.001, 'n_epochs': 3,
              'use_pseudopatients': False, 'data_source': 'umc'}

    # only top-level, fine-tune pre-trained, train from scratch

    # sweep over lr, backend, use of pseudopatients

    analysis = tune.run(train_model,
                        config={"lr": tune.grid_search([0.001, 0.01, 0.1]),
                                "backend_mode": tune.grid_search(['scratch', 'finetune','extract_only'])},
                        resources_per_trial={"gpu": 1, "cpu": 4})