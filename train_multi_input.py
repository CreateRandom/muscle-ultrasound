import random

from ignite.contrib.handlers.tensorboard_logger import WeightsScalarHandler
from torch.nn import BCELoss, BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader

from loading.loaders import make_myositis_loaders, make_umc_loader, compute_empirical_mean_and_std, \
    make_basic_transform_new, preprocess_umc_patients_new
from loading.mnist_bags import MnistBags
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
from ignite.contrib.handlers.neptune_logger import NeptuneLogger, OutputHandler, GradsScalarHandler

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

    # TRAINING ASPECTS
    batch_size = config.get('batch_size', 4)
    lr = config.get('lr', 0.001)
    n_epochs = config.get('n_epochs', 20)

    in_channels = 1 if use_one_channel else 3
    # hand over to backend
    backend_kwargs = {'pretrained': pretrained, 'feature_extraction': feature_extraction, 'in_channels': in_channels}

    data_source = config.get('data_source', 'umc')
    use_pseudopatients = config.get('use_pseudopatients', False)

    # CONFIG ASPECTS
    use_cuda = config.get('use_cuda', True) & torch.cuda.is_available()

    # whether a separate pass over the entire dataset should be done to log training set performance
    # as this incurs overhead, it can be turned off, then the results computed after each batch during the epoch
    # will be used instead
    log_training_metrics_clean = False

    npt_logger = NeptuneLogger(api_token=NEPTUNE_API_TOKEN,
                               project_name='createrandom/sandbox',
                               name='multi_input', params=config, offline_mode=False)

    # TODO refactor data loading
    current_host = socket.gethostname()
    data_path = 'data/devices/'
    if current_host == 'pop-os':
        mnt_path = '/mnt/chansey/'
    else:
        mnt_path = '/mnt/netcache/diag/'

    print(f'Using mount_path: {mnt_path}')

    attribute = 'Sex'
    # todo infer or hard code the type of problem based on the attribute
    is_classification = True

    # for each device, we have a test, val and a test set

    # the sets we need: supervised train, unsupervised train (potentially)
    # any number of validation sets

    # 'target_train': ['Philips_iU22/train']
    set_spec = {'source_train': ['ESAOTE_6100/train'],
                'val': [('ESAOTE_6100/val'), ('Philips_iU22/val')]}

    set_loaders = {}

    if data_source == 'myositis':
        normalizer_name = 'GE_Logiq_E'
        mnt_path = '/home/klux/Thesis_2/data'
        train_path = os.path.join(mnt_path, 'klaus/myositis/train.csv')
        val_path = os.path.join(mnt_path, 'klaus/myositis/val.csv')
        img_folder = os.path.join(mnt_path, 'klaus/myositis/processed_imgs')
        train_loader, val_loader = make_myositis_loaders(train_path, val_path, img_folder, use_one_channel,
                                                         normalizer_name, attribute, batch_size,
                                                         use_pseudopatients, is_classification=is_classification,
                                                         pin_memory=use_cuda)

    # for purposes of comparison
    elif data_source == 'mnist_bags':
        train_loader = DataLoader(MnistBags(target_number=9,
                                            mean_bag_length=20,
                                            var_bag_length=2,
                                            num_bag=200,
                                            seed=42,
                                            train=True),
                                  batch_size=1,
                                  shuffle=True)

        val_loader = DataLoader(MnistBags(target_number=9,
                                          mean_bag_length=10,
                                          var_bag_length=2,
                                          num_bag=50,
                                          seed=42,
                                          train=False),
                                batch_size=1,
                                shuffle=False)
    elif data_source == 'umc':
        img_folder = os.path.join(mnt_path, 'klaus/total_patients/')

        for set_name, set_paths in set_spec.items():

            loaders = []
            for set_path in set_paths:

                to_load = os.path.join(data_path, set_path)

                patients = preprocess_umc_patients_new(os.path.join(to_load, 'patients.pkl'),
                                                          os.path.join(to_load, 'records.pkl'),
                                                          os.path.join(to_load, 'images.pkl'),
                                                          attribute_to_use=attribute)

                device = set_path.split('/')[0]
                use_pseudopatient_locally = (set_name != 'val') & use_pseudopatients

                loader = make_umc_loader(patients, img_folder, use_one_channel, normalizer_name,
                                               attribute, batch_size, device, use_pseudopatients=use_pseudopatient_locally,
                                               is_classification=is_classification, pin_memory=use_cuda)

                loaders.append(loader)

            set_loaders[set_name] = loaders

    else:
        raise ValueError(f'Invalid data source : {data_source}')

    model = MultiInputNet(backend=backend, mil_pooling=mil_pooling,
                          classifier=classifier, mode=mil_mode, backend_kwargs=backend_kwargs)

    if is_classification:
        criterion = BCEWithLogitsLoss()  # BernoulliLoss
    else:
        criterion = MSELoss()

    # todo investigate / tune
    # optimizer = optim.SGD(model.parameters(), lr)
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
    for name, metric in metrics.items():
        metric.attach(trainer, name)

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
        train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, output_transform=custom_output_transform_eval)
        # enable training logging
        npt_logger.attach(train_evaluator,
                          log_handler=OutputHandler(tag="training_clean",
                                                    metric_names='all',
                                                    global_step_transform=global_step_from_engine(trainer)),
                          event_name=Events.EPOCH_COMPLETED)


    # for each validation set, create an evaluator and attach it to the logger
    val_names = set_spec['val']
    val_loaders = set_loaders['val']
    val_evaluators = []

    for name, loader in zip(val_names, val_loaders):

        val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device,
                                                    output_transform=custom_output_transform_eval)
        # enable validation logging
        npt_logger.attach(val_evaluator,
                          log_handler=OutputHandler(tag=name,
                                                    metric_names=list(metrics.keys()),
                                                    global_step_transform=global_step_from_engine(trainer)),
                          event_name=Events.EPOCH_COMPLETED)
        val_evaluators.append(val_evaluator)

    # npt_logger.attach(trainer,
    #                   log_handler=GradsScalarHandler(model),
    #                   event_name=Events.ITERATION_COMPLETED(every=1))

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
            print(train_evaluator.state.epoch, train_evaluator.state.metrics)

        # run on validation each set
        for val_evaluator, val_loader in zip(val_evaluators, val_loaders):
            val_evaluator.run(val_loader)
            print(val_evaluator.state.epoch, val_evaluator.state.metrics)

         #  tune.track.log(iter=evaluator.state.epoch, mean_accuracy=metrics['accuracy'])

    train_loader = set_loaders['source_train'][0]
    trainer.run(train_loader, max_epochs=n_epochs)

    npt_logger.close()

if __name__ == '__main__':
    # TODO read out from argparse
    config = {'backend_mode': 'extract_only',
              'backend': 'resnet-18', 'mil_pooling': 'mean', 'classifier': 'fc',
              'mil_mode': 'embedding', 'batch_size': 1, 'lr': 0.1, 'n_epochs': 1,
              'use_pseudopatients': False, 'data_source': 'umc'}

    train_model(config)
