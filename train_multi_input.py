import random

from torch.nn import BCELoss, BCEWithLogitsLoss, MSELoss

from loading.loaders import make_bag_loader, umc_to_patient_list, \
    SetSpec, get_data_for_spec, make_image_loader
from models.multi_input import MultiInputNet, BernoulliLoss
import os
import socket

import torch
from torch import nn, optim

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, Engine
from ignite.metrics import Accuracy, Loss, Precision, Recall, MeanAbsoluteError
from ignite.contrib.handlers import ProgressBar, TensorboardLogger
from ignite.handlers import ModelCheckpoint, global_step_from_engine
import numpy as np

# import logging
from ignite.contrib.handlers.neptune_logger import NeptuneLogger, OutputHandler

from models.premade import make_resnet_18
from utils.ignite_utils import PositiveShare, Variance, Average
from utils.tokens import NEPTUNE_API_TOKEN

y_epoch = []
ypred_epoch = []
losses = []


def binarize_sigmoid(y_pred):
    # the sigmoid is not part of the model
    y_sig = nn.Sigmoid()(y_pred)
    return torch.ge(y_sig, 0.5).int()


# if we have an FC as the last layer
def binarize_softmax(y_pred):
    sm = nn.Softmax(dim=1)(y_pred)
    _, i = torch.max(sm, dim=1)
    return i


def binarize_output(output):
    y_pred, y = output['y_pred'], output['y']
    y_pred = binarize_sigmoid(y_pred)
    return y_pred, y


def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_model(config):
    print(config)
    seed = config.get('seed', 42)
    fix_seed(seed)

    # BACKEND aspects
    backend_mode = config.get('backend_mode', 'scratch')

    if backend_mode == 'scratch':
        pretrained = False
        feature_extraction = False
        use_one_channel = True
        pretrained_normalizer = False
    elif backend_mode == 'finetune':
        pretrained = True
        feature_extraction = False
        use_one_channel = False
        pretrained_normalizer = True
    elif backend_mode == 'extract_only':
        pretrained = True
        feature_extraction = True
        use_one_channel = False
        pretrained_normalizer = True
    else:
        raise ValueError(f'Unrecognized backend mode: {backend_mode}')

    if pretrained_normalizer:
        normalizer_name = 'pretrained'
    else:
        normalizer_name = 'ESAOTE_6100'

    backend = config.get('backend', 'resnet-18')
    mil_pooling = config.get('mil_pooling', 'attention')
    classifier = config.get('classifier', 'fc')
    mil_mode = config.get('mil_mode', 'embedding')

    attribute = config.get('prediction_target', 'Age')

    # for now hard code the type of problem based on the attribute
    problems = {'Sex': 'binary_classification', 'Age': 'regression', 'BMI': 'regression'}
    if attribute not in problems:
        raise ValueError(f'Unknown attribute {attribute}')
    is_classification = (problems[attribute] == 'binary_classification')

    # TRAINING ASPECTS
    batch_size = config.get('batch_size', 4)

    # if backend_mode != 'extract_only' and batch_size > 2:
    #     print(f'Limiting batch size for non-extractive training.')
    #     batch_size = 2

    lr = config.get('lr', 0.001)
    n_epochs = config.get('n_epochs', 20)

    in_channels = 1 if use_one_channel else 3
    # hand over to backend
    backend_kwargs = {'pretrained': pretrained, 'feature_extraction': feature_extraction, 'in_channels': in_channels}

    use_pseudopatients = config.get('use_pseudopatients', False)

    # CONFIG ASPECTS
    use_cuda = config.get('use_cuda', True) & torch.cuda.is_available()

    # whether a separate pass over the entire dataset should be done to log training set performance
    # as this incurs overhead, it can be turned off, then the results computed after each batch during the epoch
    # will be used instead
    log_training_metrics_clean = False
    project_name = 'createrandom/mus-' + attribute
    npt_logger = NeptuneLogger(api_token=NEPTUNE_API_TOKEN,
                               project_name=project_name,
                               name='multi_input', params=config, offline_mode=True)

    current_host = socket.gethostname()
    if current_host == 'pop-os':
        mnt_path = '/mnt/chansey/'
    else:
        mnt_path = '/mnt/netcache/diag/'

    print(f'Using mount_path: {mnt_path}')

    # paths to the different datasets
    umc_data_path = os.path.join(mnt_path, 'klaus/data/devices/')
    umc_img_root = os.path.join(mnt_path, 'klaus/total_patients/')
    jhu_data_path = os.path.join(mnt_path, 'klaus/myositis/')
    jhu_img_root = os.path.join(mnt_path, 'klaus/myositis/processed_imgs')

    # for each device, we have a test, val and a test set
    device_mapping = {'ESAOTE_6100': 'umc', 'GE_Logiq_E': 'jhu', 'Philips_iU22': 'umc'}
    device_splits = {'ESAOTE_6100': ['train', 'val', 'test'], 'GE_Logiq_E': ['im_muscle_chart'],
                     'Philips_iU22': ['train', 'val', 'test']}

    label_paths = {'umc': umc_data_path, 'jhu': jhu_data_path}
    img_root_paths = {'umc': umc_img_root, 'jhu': jhu_img_root}
    set_specs = []
    for device, dataset_type in device_mapping.items():
        # get the splits
        splits = device_splits[device]
        label_path = label_paths[dataset_type]
        img_root_path = img_root_paths[dataset_type]
        for split in splits:
            set_specs.append(SetSpec(device, dataset_type, split, label_path, img_root_path))

    # the sets we need: supervised train, unsupervised train (potentially)
    # any number of validation sets

    # 'target_train': SetSpec('Philips_iU22', 'umc', 'train', umc_data_path)
    desired_set_specs = {'source_train': [SetSpec('ESAOTE_6100', 'umc', 'train', umc_data_path, umc_img_root)],
                         'val': [SetSpec('ESAOTE_6100', 'umc', 'val', umc_data_path, umc_img_root),
                                 SetSpec('Philips_iU22', 'umc', 'val', umc_data_path, umc_img_root),
                                 SetSpec('GE_Logiq_E', 'jhu', 'im_muscle_chart', jhu_data_path, jhu_img_root)]}

    set_loaders = {}

    for set_name, set_spec_list in desired_set_specs.items():

        loaders = []
        for set_spec in set_spec_list:
            data = get_data_for_spec(set_spec, loader_type='image',attribute=attribute)
            use_pseudopatient_locally = (set_name != 'val') & use_pseudopatients
            img_path = set_spec.img_root_path
            # loader = make_bag_loader(data, img_path, use_one_channel, normalizer_name,
            #                          attribute, batch_size, set_spec.device, use_pseudopatients=use_pseudopatient_locally,
            #                          is_classification=is_classification, pin_memory=use_cuda)

            loader = make_image_loader(data, img_path, use_one_channel, normalizer_name,
                                     attribute, batch_size, set_spec.device,
                                     is_classification=is_classification, pin_memory=use_cuda)

            loaders.append(loader)

        set_loaders[set_name] = loaders

    # model = MultiInputNet(backend=backend, mil_pooling=mil_pooling,
    #                       classifier=classifier, mode=mil_mode, backend_kwargs=backend_kwargs)

    model = make_resnet_18(num_classes=1, pretrained=pretrained, in_channels=in_channels,
                           feature_extraction=feature_extraction)

    if is_classification:
        criterion = BCEWithLogitsLoss()  # BernoulliLoss
    else:
        criterion = MSELoss()

    # todo investigate / tune
    optimizer = optim.Adam(model.parameters(), lr)

    device = torch.device("cuda:0" if use_cuda else "cpu")
    # needs to be manually enforced to work on the cluster
    model.to(device)

    # this custom transform allows attaching metrics directly to the trainer
    # as y and y_pred can be read out from the output dict
    def custom_output_transform(x, y, y_pred, loss):
        return {
            "y": y,
            "y_pred": y_pred,
            "loss": loss.item()
        }

    trainer = create_supervised_trainer(model, optimizer, criterion,
                                        device=device, output_transform=custom_output_transform)

    # always log the loss
    metrics = {'loss': Loss(criterion, output_transform=lambda x: (x['y_pred'], x['y']))}

    if is_classification:
        # TODO adapt for non-binary case
        metrics_to_add = {'accuracy': Accuracy(output_transform=binarize_output),
                          'p': Precision(output_transform=binarize_output),
                          'r': Recall(output_transform=binarize_output),
                          'pos_share': PositiveShare(output_transform=binarize_output)
                          }


    else:

        metrics_to_add = {'mae': MeanAbsoluteError(),
                          'mean': Average(output_transform=lambda output: output['y_pred']),
                          'var': Variance(output_transform=lambda output: output['y_pred'])}

    metrics = {**metrics, **metrics_to_add}

    # attach metrics to the trainer
    for set_spec, metric in metrics.items():
        metric.attach(trainer, set_spec)

    # there are two ways to log metrics during training
    # one can log them during the epoch, after each batch
    # and then just compute the average across batches
    # this is what e.g. standard Keras does
    # or one can compute a clean pass after all the batches have been processed, iterating over them again
    # the latter is the standard practice in ignite examples, but incurs some considerable overhead

    # attach directly to trainer (log results after each batch)
    npt_logger.attach(trainer,
                      log_handler=OutputHandler(tag="training",
                                                metric_names='all'),
                      event_name=Events.EPOCH_COMPLETED)

    def custom_output_transform_eval(x, y, y_pred):
        return {
            "y": y,
            "y_pred": y_pred,
        }

    # only if desired incur the extra overhead
    if log_training_metrics_clean:
        # make a separate evaluator and attach to it instead (do a clean pass)
        train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device,
                                                      output_transform=custom_output_transform_eval)
        # enable training logging
        npt_logger.attach(train_evaluator,
                          log_handler=OutputHandler(tag="training_clean",
                                                    metric_names='all',
                                                    global_step_transform=global_step_from_engine(trainer)),
                          event_name=Events.EPOCH_COMPLETED)

    # for each validation set, create an evaluator and attach it to the logger
    val_names = desired_set_specs['val']
    val_loaders = set_loaders['val']
    val_evaluators = []

    for set_spec, loader in zip(val_names, val_loaders):
        val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device,
                                                    output_transform=custom_output_transform_eval)
        # enable validation logging
        npt_logger.attach(val_evaluator,
                          log_handler=OutputHandler(tag=str(set_spec),
                                                    metric_names=list(metrics.keys()),
                                                    global_step_transform=global_step_from_engine(trainer)),
                          event_name=Events.EPOCH_COMPLETED)
        val_evaluators.append(val_evaluator)

    # todo migrate this to tensorboard
    # npt_logger.attach(trainer,
    #                    log_handler=GradsScalarHandler(model),
    #                    event_name=Events.EPOCH_COMPLETED)
    #
    # npt_logger.attach(trainer,
    #                    log_handler=WeightsScalarHandler(model),
    #                    event_name=Events.EPOCH_COMPLETED)
    #

    pbar = ProgressBar()
    pbar.attach(trainer)

    checkpoint_dir = 'checkpoints'

    # checkpointer = ModelCheckpoint(checkpoint_dir, 'pref', n_saved=3, create_dir=True, require_empty=False)
    # trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), checkpointer, {'mymodel': model})

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(trainer):
        print(trainer.state.epoch, trainer.state.metrics)
        if log_training_metrics_clean:
            train_evaluator.run(train_loader)
            print(trainer.state.epoch, train_evaluator.state.metrics)

        # run on validation each set
        for val_evaluator, val_loader in zip(val_evaluators, val_loaders):
            val_evaluator.run(val_loader)
            print(trainer.state.epoch, val_evaluator.state.metrics)

        #  tune.track.log(iter=evaluator.state.epoch, mean_accuracy=metrics['accuracy'])

    train_loader = set_loaders['source_train'][0]
    trainer.run(train_loader, max_epochs=n_epochs)

    npt_logger.close()


if __name__ == '__main__':
    # TODO read out from argparse
    config = {'prediction_target': 'Age', 'backend_mode': 'scratch',
              'backend': 'resnet-18', 'mil_pooling': 'mean', 'classifier': 'fc',
              'mil_mode': 'embedding', 'batch_size': 8, 'lr': 0.1, 'n_epochs': 5,
              'use_pseudopatients': False}

    train_model(config)
