import random

from sklearn.metrics import mean_absolute_error, accuracy_score
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss

from loading.datasets import CustomLabelEncoder
from loading.loaders import make_bag_loader, \
    SetSpec, get_data_for_spec, make_image_loader, get_classes
from models.multi_input import MultiInputNet, cnn_constructors
import os
import socket

import torch
from torch import optim

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall, MeanAbsoluteError, MetricsLambda, ConfusionMatrix
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import global_step_from_engine
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor

# import logging
from ignite.contrib.handlers.neptune_logger import NeptuneLogger, OutputHandler, WeightsScalarHandler, \
    GradsScalarHandler

from utils.ignite_utils import PositiveShare, Variance, Average, binarize_softmax, binarize_sigmoid
from utils.tokens import NEPTUNE_API_TOKEN

y_epoch = []
ypred_epoch = []
losses = []


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

    # image level vs bag level
    problem_type = config.get('problem_type', 'bag')

    backend = config.get('backend', 'resnet-18')
    mil_pooling = config.get('mil_pooling', 'attention')

    fc_hidden_layers = config.get('fc_hidden_layers', 0)
    fc_use_bn = config.get('fc_use_bn', True)
    mil_mode = config.get('mil_mode', 'embedding')

    attribute = config.get('prediction_target', 'Age')

    # for now hard code the type of problem based on the attribute
    problems = {'Sex': 'binary', 'Age': 'regression', 'BMI': 'regression',
                'Muscle': 'multi', 'Class': 'multi'}
    if attribute not in problems:
        raise ValueError(f'Unknown attribute {attribute}')
    is_classification = (problems[attribute] == 'multi' or (problems[attribute] == 'binary'))
    is_multi = (problems[attribute] == 'multi')
    # TRAINING ASPECTS
    batch_size = config.get('batch_size', 4)
    # more than two bags per batch are likely to overwhelm memory
    if backend_mode != 'extract_only' and problem_type == 'bag' and batch_size > 2:
        print(f'Limiting batch size for non-extractive training.')
        batch_size = 2

    # whether to crop images to ImageNet size (i.e. 224 * 224)
    limit_image_size = config.get('limit_image_size', True)
    lr = config.get('lr', 0.001)
    n_epochs = config.get('n_epochs', 20)

    in_channels = 1 if use_one_channel else 3
    # hand over to backend
    backend_kwargs = {'pretrained': pretrained, 'feature_extraction': feature_extraction, 'in_channels': in_channels}

    use_pseudopatients = config.get('use_pseudopatients', False)

    # CONFIG ASPECTS
    use_cuda = config.get('use_cuda', True) & torch.cuda.is_available()

    # change logging and data location based on machine
    current_host = socket.gethostname()
    offline_mode = False
    if current_host == 'pop-os':
        mnt_path = '/mnt/chansey/'
        offline_mode = True
    else:
        mnt_path = '/mnt/netcache/diag/'

    print(f'Using mount_path: {mnt_path}')

    # whether a separate pass over the entire dataset should be done to log training set performance
    # as this incurs overhead, it can be turned off, then the results computed after each batch during the epoch
    # will be used instead
    log_training_metrics_clean = False
    log_grads = False
    log_weights = False
    project_name = 'createrandom/mus-base'
    npt_logger = NeptuneLogger(api_token=NEPTUNE_API_TOKEN, project_name=project_name,
                               tags=[attribute, problem_type],
                               name='multi_input', params=config, offline_mode=offline_mode)

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

    label_encoder = None
    train_classes = None

    muscles_to_use = ['Biceps brachii', 'Tibialis anterior', 'Gastrocnemius medial head', 'Flexor carpi radialis',
                      'Vastus lateralis']

    for set_name, set_spec_list in desired_set_specs.items():

        loaders = []
        for set_spec in set_spec_list:
            print(set_spec)
            # pass the classes in to ensure that only those are present in all the sets
            data = get_data_for_spec(set_spec, loader_type=problem_type, attribute=attribute,
                                     class_values=train_classes,
                                     muscles_to_use=muscles_to_use)
            print(f'Loaded {len(data)} elements.')

            # if classification and this is the train set, we want to fit the label encoder on this
            if is_classification & (set_name == 'source_train'):
                train_classes = get_classes(data, attribute)
                print(train_classes)
                label_encoder = CustomLabelEncoder(train_classes, one_hot_encode=False)
            print(get_classes(data, attribute))
            img_path = set_spec.img_root_path
            # decide which type of loader we need here
            if problem_type == 'bag':
                use_pseudopatient_locally = (set_name != 'val') & use_pseudopatients
                loader = make_bag_loader(data, img_path, use_one_channel, normalizer_name,
                                         attribute, batch_size, set_spec.device, limit_image_size=limit_image_size,
                                         use_pseudopatients=use_pseudopatient_locally,
                                         label_encoder=label_encoder, pin_memory=use_cuda)
            elif problem_type == 'image':
                loader = make_image_loader(data, img_path, use_one_channel, normalizer_name,
                                           attribute, batch_size, set_spec.device, limit_image_size=limit_image_size,
                                           label_encoder=label_encoder, pin_memory=use_cuda, is_multi=is_multi)
            else:
                raise ValueError(f'Unknown problem type {problem_type}')

            loaders.append(loader)

        set_loaders[set_name] = loaders

    # output dimensionality of the network
    if train_classes and (len(train_classes) > 2):
        num_classes = len(train_classes)
    # regression or binary classification with sigmoid
    else:
        num_classes = 1

    # obtain baseline performance estimates

    print('Baseline scores')
    train_labels = set_loaders['source_train'][0].dataset.get_all_labels()
    if is_classification:
        d = DummyClassifier(strategy='most_frequent')
        scorer = accuracy_score
    else:
        d = DummyRegressor(strategy='mean')
        scorer = mean_absolute_error
    d.fit([0] * len(train_labels), train_labels)
    train_preds = d.predict([0] * len(train_labels))
    score = scorer(train_labels, train_preds)
    print(score)

    for val_loader in set_loaders['val']:
        val_labels = val_loader.dataset.get_all_labels()
        val_preds = d.predict([0] * len(val_labels))
        score = scorer(val_labels, val_preds)
        print(score)

    # decide which type of model we want to train
    if problem_type == 'bag':
        model = MultiInputNet(backend=backend, mil_pooling=mil_pooling,
                              mode=mil_mode, out_dim=num_classes,
                              fc_hidden_layers=fc_hidden_layers, fc_use_bn=fc_use_bn,
                              backend_kwargs=backend_kwargs)

        # can specify different learning rates for each component!
        # optimizer = optim.Adam([{'params': model.backend.parameters(), 'lr': 0},
        #                         {'params': model.classifier.parameters(), 'lr': lr}], lr)

        optimizer = optim.Adam(model.parameters(), lr)

    elif problem_type == 'image':
        # get the right cnn
        backend_func = cnn_constructors[backend]
        # add the output dim
        backend_kwargs['num_classes'] = num_classes
        model = backend_func(**backend_kwargs)
        # train the whole backend with the same lr
        optimizer = optim.Adam(model.parameters(), lr)
    else:
        raise ValueError(f'Unknown problem type {problem_type}')
    print(f'Using {model}')

    if is_classification:
        if num_classes == 1:
            criterion = BCEWithLogitsLoss()  # BernoulliLoss
        else:
            criterion = CrossEntropyLoss()
    else:
        criterion = MSELoss()

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

    # binary cases
    if is_classification & (num_classes == 1):
        metrics_to_add = {'accuracy': Accuracy(output_transform=binarize_sigmoid),
                          'p': Precision(output_transform=binarize_sigmoid),
                          'r': Recall(output_transform=binarize_sigmoid),
                          'pos_share': PositiveShare(output_transform=binarize_sigmoid)
                          }
    # non-binary case
    elif is_classification:
        p = Precision(output_transform=binarize_softmax, average=False)
        r = Recall(output_transform=binarize_softmax, average=False)
        F1 = p * r * 2 / (p + r + 1e-20)
        F1 = MetricsLambda(lambda t: torch.mean(t).item(), F1)

        metrics_to_add = {'accuracy': Accuracy(output_transform=binarize_softmax),
                          'ap': Precision(output_transform=binarize_softmax, average=True),
                          'ar': Recall(output_transform=binarize_softmax, average=True),
                          'f1': F1,
                          'cm': ConfusionMatrix(output_transform=binarize_softmax, num_classes=num_classes)
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
    if log_grads:
        npt_logger.attach(trainer,
                          log_handler=GradsScalarHandler(model),
                          event_name=Events.ITERATION_COMPLETED)
    if log_weights:
        npt_logger.attach(trainer,
                          log_handler=WeightsScalarHandler(model),
                          event_name=Events.ITERATION_COMPLETED)

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
    config = {'problem_type': 'bag', 'prediction_target': 'Sex', 'backend_mode': 'finetune',
              'backend': 'resnet-18', 'mil_pooling': 'mean',
              'mil_mode': 'embedding', 'batch_size': 2, 'lr': 0.0269311, 'n_epochs': 5,
              'use_pseudopatients': False, 'fc_hidden_layers': 0, 'fc_use_bn': True}

    train_model(config)
