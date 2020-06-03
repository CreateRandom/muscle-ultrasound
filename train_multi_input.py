import random

from ignite.contrib.handlers.tensorboard_logger import WeightsScalarHandler
from sklearn.model_selection import train_test_split
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
from ignite.metrics import Accuracy, Loss, Precision, Recall, MeanSquaredError, MeanAbsoluteError
from ignite.contrib.handlers import ProgressBar, TensorboardLogger
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from sklearn.metrics import accuracy_score, precision_score, recall_score
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
    y_pred, y = output
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
    use_cuda = config.get('use_cuda', True) & torch.cuda.is_available()

    npt_logger = NeptuneLogger(api_token=NEPTUNE_API_TOKEN,
                               project_name='createrandom/muscle-ultrasound',
                               name='multi_input', params=config, offline_mode=True)

    # TODO refactor data loading
    current_host = socket.gethostname()
    data_path = 'data/devices/'
    if current_host == 'pop-os':
        mnt_path = '/mnt/chansey/'
    else:
        mnt_path = '/mnt/netcache/diag/'

    print(f'Using mount_path: {mnt_path}')

    attribute = 'Age'
    # todo infer or hard code the type of problem based on the attribute
    is_classification = False

    # for each device, we have a test, val and a test set

    # the sets we need: supervised train, unsupervised train (potentially)
    # any number of validation sets

    set_spec = {'source_train': ['ESAOTE_6100/train'], 'target_train': ['Philips_iU22/train'],
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

    # called on every iteration
    def evaluate_training_items(x, y, y_pred, loss):
        preds = binarize_sigmoid(y_pred)
        ypred_epoch.extend(preds.cpu().numpy().tolist())
        y_epoch.extend(y.cpu().numpy().tolist())
        losses.append(loss.item())
        return loss.item()

    trainer = create_supervised_trainer(model, optimizer, criterion,
                                        device=device)#, output_transform=evaluate_training_items)

    # always log the loss
    metrics = {'loss': Loss(criterion)}

    if is_classification:
        # TODO adapt for non-binary case
        metrics_to_add = {'accuracy': Accuracy(output_transform=binarize_output),
                          'p': Precision(output_transform=binarize_output),
                          'r': Recall(output_transform=binarize_output),
                          'pos_share': PositiveShare(output_transform=binarize_output)
                          }


    else:
        def extract_pred(output):
            y_pred, y = output
            return y_pred

        metrics_to_add = {'mse': MeanSquaredError(), 'mae': MeanAbsoluteError(),
                          'mean': Average(output_transform=extract_pred),
                          'var': Variance(output_transform=extract_pred)}

    metrics = {**metrics, **metrics_to_add}

    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    # enable training logging
    npt_logger.attach(train_evaluator,
                      log_handler=OutputHandler(tag="training",
                                                metric_names=list(metrics.keys()),
                                                global_step_transform=global_step_from_engine(trainer)),
                      event_name=Events.EPOCH_COMPLETED)


    # for each validation set, create an evaluator and attach it to the logger
    val_names = set_spec['val']
    val_loaders = set_loaders['val']
    val_evaluators = []
    for name, loader in zip(val_names, val_loaders):

        val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
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
    def log_training_results(trainer):
        # this is a waste of resources
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        print(trainer.state.epoch)
        print(metrics)
        # print(
        #     "Training Results - Epoch: {}  Avg accuracy: {:.2f} "
        #     "Avg precision: {:.2f} Avg recall: {:.2f} Avg loss: {:.2f} Share positive: {:.2f}"
        #         .format(trainer.state.epoch, metrics['accuracy'], metrics['p'], metrics['r'],
        #                 metrics['loss'], metrics['pos_share']))

        # a more convoluted but faster way (saves the double execution cost)

        # global losses, y_epoch, ypred_epoch
        # loss = np.mean(losses)
        # acc = accuracy_score(y_epoch, ypred_epoch)
        # p = precision_score(y_epoch, ypred_epoch)
        # r = recall_score(y_epoch, ypred_epoch)
        # print(
        #     "Training Results - Epoch: {} Avg accuracy: {:.2f} Avg precision: {:.2f} Avg recall: {:.2f} Avg loss: {:.2f}"
        #         .format(trainer.state.epoch, acc, p, r, loss))
        #
        # # overwrite to enable logging
        # train_evaluator.state.metrics = {'accuracy': acc, 'p': p, 'r': r, 'loss': loss}
        #
        # # reset the storage
        # y_epoch = []
        # ypred_epoch = []
        # losses = []

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        # run on each set
        for val_evaluator, val_loader in zip(val_evaluators, val_loaders):
            val_evaluator.run(val_loader)
            metrics = val_evaluator.state.metrics
            print(trainer.state.epoch)
            print(metrics)
        # print(
        #     "Validation Results - Epoch: {}  Avg accuracy: {:.2f} "
        #     "Avg precision: {:.2f} Avg recall: {:.2f} Avg loss: {:.2f} Share positive: {:.2f}"
        #     .format(trainer.state.epoch, metrics['accuracy'], metrics['p'], metrics['r'], metrics['loss'], metrics['pos_share']))

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
