import random

from sklearn.metrics import mean_absolute_error, accuracy_score
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss

from loading.datasets import CustomLabelEncoder
from loading.loaders import make_bag_loader, \
    SetSpec, get_data_for_spec, make_image_loader, get_classes
from loading.loading_utils import make_set_specs
from models.multi_input import MultiInputNet, cnn_constructors, MultiInputBaseline
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
from training_utils import problems, fix_seed
# import logging
from ignite.contrib.handlers.neptune_logger import NeptuneLogger, OutputHandler, WeightsScalarHandler, \
    GradsScalarHandler

from utils.ignite_utils import PositiveShare, Variance, Average, binarize_softmax, binarize_sigmoid, \
    pytorch_count_params, create_bag_attention_trainer, create_bag_attention_evaluator, Minimum, Maximum
from utils.tokens import NEPTUNE_API_TOKEN

def train_multi_input(config):
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
    mil_pooling = config.get('mil_pooling', 'mean')
    attention_mode = config.get('attention_mode', 'identity')
    attention_D = config.get('attention_D', 128)
    pooling_kwargs = {'mode' : attention_mode, 'D': attention_D}

    fc_hidden_layers = config.get('fc_hidden_layers', 0)
    fc_use_bn = config.get('fc_use_bn', True)
    # how many layers of the backend to chop of from the bottom
    backend_cutoff = config.get('backend_cutoff', 0)
    mil_mode = config.get('mil_mode', 'embedding')

    attribute = config.get('prediction_target', 'Age')

    if attribute not in problems:
        raise ValueError(f'Unknown attribute {attribute}')
    label_type = problems[attribute]
    is_classification = (label_type == 'multi' or (label_type == 'binary'))
    is_multi = (label_type == 'multi')
    # TRAINING ASPECTS
    batch_size = config.get('batch_size', 4)
    # more than two bags per batch are likely to overwhelm memory

    # whether to crop images to ImageNet size (i.e. 224 * 224)
    limit_image_size = config.get('limit_image_size', True)
    lr = config.get('lr', 0.001)
    # separate lr for the backend can be specified, defaults to normal LR
    backend_lr = config.get('backend_lr', lr)

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
    project_name = config.get('neptune_project','createrandom/muscle-ultrasound')
    if 'neptune_project' in config:
        config.pop('neptune_project')
    # paths to the different datasets
    umc_data_path = os.path.join(mnt_path, 'klaus/data/devices/')
    umc_img_root = os.path.join(mnt_path, 'klaus/total_patients/')
    jhu_data_path = os.path.join(mnt_path, 'klaus/myositis/')
    jhu_img_root = os.path.join(mnt_path, 'klaus/myositis/processed_imgs')

    # yields a mapping from names to set_specs
    set_spec_dict = make_set_specs(umc_data_path, umc_img_root, jhu_data_path, jhu_img_root)
    # this is always needed
    source_train = config.get('source_train')
    target_train = config.get('target_train', None)
    val = config.get('val')

    # 'target_train': SetSpec('Philips_iU22', 'umc', 'train', umc_data_path)
    desired_set_specs = {'source_train': source_train, 'target_train' : target_train,
                         'val': val} # 'GE_Logiq_E_im_muscle_chart']}

    # resolve the set names to set_specs
    resolved_set_specs = {}
    for key in desired_set_specs.keys():
        elem = desired_set_specs[key]
        if not elem:
            continue
        if isinstance(elem,list):
            new_elem = []
            for name in elem:
                spec = set_spec_dict[name]
                new_elem.append(spec)
        else:
            new_elem = [set_spec_dict[elem]]
        resolved_set_specs[key] = new_elem

    desired_set_specs = resolved_set_specs

    set_loaders = {}
    label_encoder = None
    train_classes = None
    muscles_to_use = None
    use_most_frequent_muscles = config.get('muscle_subset', False)
    if use_most_frequent_muscles:
        muscles_to_use = ['Biceps brachii', 'Tibialis anterior', 'Gastrocnemius medial head', 'Flexor carpi radialis',
                          'Vastus lateralis']

    for set_name, set_spec_list in desired_set_specs.items():

        bag_loaders = []
        for set_spec in set_spec_list:
            print(set_spec)
            # pass the classes in to ensure that only those are present in all the sets
            patients = get_data_for_spec(set_spec, loader_type='bag', attribute=attribute,
                                     class_values=train_classes,
                                     muscles_to_use=muscles_to_use)
           # patients = patients[0:10]
            print(f'Loaded {len(patients)} elements.')

            # if classification and this is the train set, we want to fit the label encoder on this
            if is_classification & (set_name == 'source_train'):
                train_classes = get_classes(patients, attribute)
                print(train_classes)
                label_encoder = CustomLabelEncoder(train_classes, one_hot_encode=False)
            print(get_classes(patients, attribute))
            img_path = set_spec.img_root_path
            # decide which type of loader we need here
            # always make the bag loader
            use_pseudopatient_locally = (set_name != 'val') & use_pseudopatients
            loader = make_bag_loader(patients, img_path, use_one_channel, normalizer_name,
                                     attribute, batch_size, set_spec.device, limit_image_size=limit_image_size,
                                     use_pseudopatients=use_pseudopatient_locally,
                                     label_encoder=label_encoder, pin_memory=use_cuda)
            bag_loaders.append(loader)


        set_loaders[set_name] = bag_loaders

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

    # create the desired model
    model = MultiInputNet(backend=backend, mil_pooling=mil_pooling,
                      pooling_kwargs=pooling_kwargs,
                      mode=mil_mode, out_dim=num_classes,
                      fc_hidden_layers=fc_hidden_layers, fc_use_bn=fc_use_bn,
                      backend_cutoff=backend_cutoff,
                      backend_kwargs=backend_kwargs)

    n_params_backend = pytorch_count_params(model.backend)
    n_params_classifier = pytorch_count_params(model.classifier)
    n_params_pooling = pytorch_count_params(model.mil_pooling)
    config['n_params_classifier'] = n_params_classifier
    config['n_params_pooling'] = n_params_pooling

    # can specify different learning rates for each component!
    optimizer = optim.Adam([{'params': model.backend.parameters(), 'lr': backend_lr},
                            {'params': model.classifier.parameters(), 'lr': lr}], lr)

    config['n_params_backend'] = n_params_backend

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
            "y_pred": y_pred['preds'],
            "atts": y_pred['atts'],
            "loss": loss.item()
        }


    trainer = create_bag_attention_trainer(model, optimizer, criterion,
                                           device=device, output_transform=custom_output_transform)


    # always log the loss
    metrics = {'loss': Loss(criterion, output_transform=lambda x: (x['y_pred'], x['y']))}

    # logging attention distributions
    log_attention = mil_pooling == 'attention'
    if log_attention:
        metrics_to_add = {'mean_att': Average(output_transform=lambda output: output['atts']),
                          'var_att': Variance(output_transform=lambda output: output['atts']),
                          'min_att': Minimum(output_transform=lambda output: output['atts']),
                          'max_att': Maximum(output_transform=lambda output: output['atts'])}
        metrics = {**metrics, **metrics_to_add}
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

    npt_logger = NeptuneLogger(api_token=NEPTUNE_API_TOKEN, project_name=project_name,
                               tags=[attribute, 'bag'],
                               name='multi_input', params=config, offline_mode=offline_mode)


    # there are two ways to log metrics during training
    # one can log them during the epoch, after each batch
    # and then just compute the average across batches
    # this is what e.g. standard Keras does
    # or one can compute a clean pass after all the batches have been processed, iterating over them again
    # the latter is the standard practice in ignite examples, but incurs some considerable overhead

    # attach directly to trainer (log results after each epoch)
    npt_logger.attach(trainer,
                      log_handler=OutputHandler(tag="training",
                                                metric_names='all'),
                      event_name=Events.EPOCH_COMPLETED)

    def custom_output_transform_eval(x, y, y_pred):
        return {
            "y": y,
            "y_pred": y_pred['preds'],
            "atts": y_pred['atts'],
        }


    # only if desired incur the extra overhead
    if log_training_metrics_clean:

        # make a separate evaluator and attach to it instead (do a clean pass)
        train_evaluator = create_bag_attention_evaluator(model, metrics=metrics, device=device,
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
        eval_name = str(set_spec)
        val_evaluator = create_bag_attention_evaluator(model, metrics=metrics, device=device,
                                                       output_transform=custom_output_transform_eval)

        # enable validation logging
        npt_logger.attach(val_evaluator,
                          log_handler=OutputHandler(tag=eval_name,
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

    # get the appropriate loader
    train_loader = set_loaders['source_train'][0]
    trainer.run(train_loader, max_epochs=n_epochs)

    npt_logger.close()

if __name__ == '__main__':
    # TODO read out from argparse
    bag_config = {'problem_type': 'bag', 'prediction_target': 'Sex', 'backend_mode': 'finetune',
                  'backend': 'resnet-18', 'mil_pooling': 'attention', 'attention_mode': 'sigmoid',
                  'mil_mode': 'embedding', 'batch_size': 4, 'lr': 0.0269311, 'n_epochs': 5,
                  'use_pseudopatients': True, 'fc_hidden_layers': 2, 'fc_use_bn': True,
                  'backend_cutoff': 1}

    only_philips = {'source_train': 'Philips_iU22_train',
                         'val': 'Philips_iU22_val'}

    config = {**bag_config, **only_philips}

    train_multi_input(config)
