from training_utils import fix_seed
from loading.datasets import problem_kind, make_att_specs
from loading.loaders import make_bag_loader, \
    get_data_for_spec, make_image_loader, get_classes
from loading.loading_utils import make_set_specs
from models.multi_input import cnn_constructors, MultiInputBaseline
import os
import socket

import torch
from torch import optim

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Precision, Recall, MeanAbsoluteError
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import global_step_from_engine, ModelCheckpoint

# import logging
from ignite.contrib.handlers.neptune_logger import NeptuneLogger, OutputHandler, WeightsScalarHandler, \
    GradsScalarHandler

from utils.binarize_utils import binarize_sigmoid, apply_sigmoid
from utils.trainers import StateDictWrapper
from utils.utils import pytorch_count_params
from utils.ignite_metrics import PositiveShare, Variance, Average, loss_mapping, obtain_metrics, AUC, \
    default_metric_mapping
from utils.tokens import NEPTUNE_API_TOKEN


def train_image_level(config):
    print(config)
    config['problem_type'] = 'image'
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

    attribute = config.get('prediction_target', 'Age')

    if attribute not in problem_kind:
        raise ValueError(f'Unknown attribute {attribute}')
    att_spec_dict = make_att_specs()
    attribute_specs = [att_spec_dict[attribute]]

    label_type = problem_kind[attribute]
    config['label_type'] = label_type
    is_multi = (label_type == 'multi')
    # TRAINING ASPECTS
    batch_size = config.get('batch_size', 4)
    patient_batch_size = config.get('patient_batch_size', 4)

    # whether to crop images to ImageNet size (i.e. 224 * 224)
    limit_image_size = config.get('limit_image_size', True)
    lr = config.get('lr', 0.001)

    n_epochs = config.get('n_epochs', 20)

    in_channels = 1 if use_one_channel else 3
    config['in_channels'] = in_channels
    # hand over to backend
    backend_kwargs = {'pretrained': pretrained, 'feature_extraction': feature_extraction, 'in_channels': in_channels}

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
    project_name = config.get('neptune_project', 'createrandom/muscle-ultrasound')
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
    desired_set_specs = {'source_train': source_train, 'target_train': target_train,
                         'val': val}  # 'GE_Logiq_E_im_muscle_chart']}

    # resolve the set names to set_specs
    resolved_set_specs = {}
    for key in desired_set_specs.keys():
        elem = desired_set_specs[key]
        if not elem:
            continue
        if isinstance(elem, list):
            new_elem = []
            for name in elem:
                spec = set_spec_dict[name]
                new_elem.append(spec)
        else:
            new_elem = [set_spec_dict[elem]]
        resolved_set_specs[key] = new_elem

    desired_set_specs = resolved_set_specs

    set_loaders = {}
    set_loaders['bag'] = {}
    set_loaders['image'] = {}

    train_classes = att_spec_dict[attribute].legal_values
   # filter_attribute = 'Class_sample' if attribute == 'Class' else None

    muscles_to_use = None
    use_most_frequent_muscles = config.get('muscle_subset', False)
    if use_most_frequent_muscles:
        muscles_to_use = ['Biceps brachii', 'Tibialis anterior', 'Gastrocnemius medial head', 'Flexor carpi radialis',
                          'Vastus lateralis']

    train_transform_params = None
    for set_name, set_spec_list in desired_set_specs.items():

        bag_loaders = []
        image_loaders = []
        for set_spec in set_spec_list:
            print(set_spec)

            # pass the classes in to ensure that only those are present in all the sets
            patients = get_data_for_spec(set_spec, loader_type='bag', attribute_to_filter=attribute,
                                         legal_attribute_values=train_classes,
                                         muscles_to_use=muscles_to_use,
                                         dropna_values=True)

          #  patients = patients[0:10]
            print(f'Loaded {len(patients)} elements.')

            img_path = set_spec.img_root_path

            # make the bag loader
            loader = make_bag_loader(patients, img_path, use_one_channel, normalizer_name,
                                     attribute_specs, patient_batch_size, set_spec.device, limit_image_size=limit_image_size,
                                     use_pseudopatients=False, return_attribute_dict= False,
                                     pin_memory=use_cuda)
            bag_loaders.append(loader)

            # make the image loader
            images = get_data_for_spec(set_spec, loader_type='image', attribute_to_filter=attribute,
                                       legal_attribute_values=train_classes,
                                       muscles_to_use=muscles_to_use,
                                       dropna_values=True)
        #    images = images.sample(n=250)
            
            image_loader = make_image_loader(images, img_path, use_one_channel, normalizer_name,
                                             attribute_specs, batch_size, set_spec.device, limit_image_size=limit_image_size,
                                             pin_memory=use_cuda, is_multi=is_multi, return_multiple_atts= False)
            image_loaders.append(image_loader)

            transform_params = {'resize_option_name' : set_spec.device, 'normalizer_name': normalizer_name,
                                'limit_image_size': limit_image_size}

            # store the transform params used for training
            if set_name == 'source_train':
                train_transform_params = transform_params

        set_loaders['bag'][set_name] = bag_loaders
        set_loaders['image'][set_name] = image_loaders

    # output dimensionality of the network
    if train_classes and (len(train_classes) > 2):
        num_classes = len(train_classes)
    # regression or binary classification with sigmoid
    else:
        num_classes = 1

    # get the right cnn
    backend_func = cnn_constructors[backend]
    # add the output dim
    backend_kwargs['num_classes'] = num_classes
    model = backend_func(**backend_kwargs)
    n_params_backend = pytorch_count_params(model)
    # train the whole backend with the same lr
    optimizer = optim.Adam(model.parameters(), lr)
    patient_eval = MultiInputBaseline(model, label_type=label_type)

    print(f'Using {model}')

    config['n_params_backend'] = n_params_backend

    criterion = loss_mapping[att_spec_dict[attribute].target_type]

    device = torch.device("cuda:0" if use_cuda else "cpu")
    # needs to be manually enforced to work on the cluster
    model.to(device)
    patient_eval.to(device)

    # this custom transform allows attaching metrics directly to the trainer
    # as y and y_pred can be read out from the output dict
    def custom_output_transform(x, y, y_pred, loss):
        return {
            "y": y,
            "y_pred": y_pred,
            "loss": loss.item()
        }

    trainer = create_supervised_trainer(model, optimizer, criterion,
                                        device=device, output_transform=custom_output_transform,
                                        deterministic=True)

    metrics = obtain_metrics(attribute_specs, extract_multiple_atts=False)

    # attach metrics to the trainer
    for set_spec, metric in metrics.items():
        metric.attach(trainer, set_spec)

    npt_logger = NeptuneLogger(api_token=NEPTUNE_API_TOKEN, project_name=project_name,
                               tags=[attribute, 'image'],
                               name='image_baseline', params=config, offline_mode=offline_mode)

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
            "y_pred": y_pred,
        }

    # only if desired incur the extra overhead
    if log_training_metrics_clean:
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
    val_loaders = set_loaders['image']['val']
    val_evaluators = []

    for set_spec, loader in zip(val_names, val_loaders):
        eval_name = str(set_spec) + '_image'
        val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device,
                                                    output_transform=custom_output_transform_eval)
        # enable validation logging
        npt_logger.attach(val_evaluator,
                          log_handler=OutputHandler(tag=eval_name,
                                                    metric_names=list(metrics.keys()),
                                                    global_step_transform=global_step_from_engine(trainer)),
                          event_name=Events.EPOCH_COMPLETED)
        val_evaluators.append(val_evaluator)

    # add patient level baseline evaluators for the image level

    bag_metric_mapping = {'regression': [('mae', MeanAbsoluteError, None, {}),
                                         ('mean', Average, lambda output: output['y_pred'], {}),
                                         ('var', Variance, lambda output: output['y_pred'], {})], 'multi': [
        ('accuracy', Accuracy, None, {}),
        ('ap', Precision, None, {'average': True}),
        ('ar', Recall, None, {'average': True})
    ], 'binary': default_metric_mapping['binary']}

    bag_level_metrics = obtain_metrics(attribute_specs, extract_multiple_atts=False, metric_mapping=bag_metric_mapping,
                                       add_loss=False)

    val_loaders_bag = set_loaders['bag']['val']
    val_evaluators_bag = []
    for set_spec, loader in zip(val_names, val_loaders_bag):
        val_evaluator = create_supervised_evaluator(patient_eval, metrics=bag_level_metrics, device=device,
                                                    output_transform=custom_output_transform_eval)

        # val_evaluator = create_image_to_bag_evaluator(model, metrics=metrics_to_add, device=device,
        #                                              output_transform=custom_output_transform_eval)
        # enable validation logging
        npt_logger.attach(val_evaluator,
                          log_handler=OutputHandler(tag=str(set_spec),
                                                    metric_names=list(bag_level_metrics.keys()),
                                                    global_step_transform=global_step_from_engine(trainer)),
                          event_name=Events.EPOCH_COMPLETED)
        val_evaluators_bag.append(val_evaluator)

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

    # make checkpoint dir based on contents of config
    try:
        exp_id = npt_logger.get_experiment()._id
        checkpoint_dir = os.path.join('checkpoints', project_name, exp_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
    except:
        checkpoint_dir = 'checkpoints'

    checkpointer = ModelCheckpoint(checkpoint_dir, 'pref', create_dir=True, require_empty=False, n_saved=None,
                                   global_step_transform=global_step_from_engine(trainer))
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), checkpointer,
                              {'model_state_dict': model, 'model_param_dict': StateDictWrapper(backend_kwargs),
                               'transform_dict': StateDictWrapper(train_transform_params),
                               'config': StateDictWrapper(config)})

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

        # run the patient level baseline
        for val_evaluator, val_loader in zip(val_evaluators_bag, val_loaders_bag):
            val_evaluator.run(val_loader)
            print(trainer.state.epoch, val_evaluator.state.metrics)

        #  tune.track.log(iter=evaluator.state.epoch, mean_accuracy=metrics['accuracy'])

    # get the appropriate loader
    train_loader = set_loaders['image']['source_train'][0]
    trainer.run(train_loader, max_epochs=n_epochs)

    npt_logger.close()


if __name__ == '__main__':
    config = {'prediction_target': 'Class', 'backend_mode': 'finetune',
              'backend': 'resnet-18', 'batch_size': 32, 'lr': 0.0269311, 'n_epochs': 5}

    esoate_train = {'source_train': 'ESAOTE_6100_train',
                         'val': ['ESAOTE_6100_val']}

    config = {**config, **esoate_train}

    train_image_level(config)
