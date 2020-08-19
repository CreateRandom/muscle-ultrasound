import warnings

from loading.datasets import problem_kind, make_att_specs, ConcatDataset
from loading.loaders import make_bag_loader, get_data_for_spec, get_classes, make_bag_dataset, wrap_in_bag_loader, \
    make_basic_transform
from loading.loading_utils import make_set_specs
from models.multi_input import MultiInputNet
import os
import socket

import torch
from torch import optim

from ignite.engine import Events
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import global_step_from_engine, ModelCheckpoint

from training_utils import fix_seed
# import logging
from ignite.contrib.handlers.neptune_logger import NeptuneLogger, OutputHandler, WeightsScalarHandler, \
    GradsScalarHandler

from utils.trainers import create_bag_attention_trainer, create_bag_attention_evaluator, create_da_trainer, \
    StateDictWrapper
from utils.utils import pytorch_count_params
from utils.ignite_metrics import Variance, Average, Minimum, Maximum, obtain_metrics
from utils.tokens import NEPTUNE_API_TOKEN

def train_multi_input(config):
    print(config)
    config['problem_type'] = 'bag'
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
    pooling_kwargs = {'mode': attention_mode, 'D': attention_D}

    fc_hidden_layers = config.get('fc_hidden_layers', 0)
    fc_use_bn = config.get('fc_use_bn', True)
    # how many layers of the backend to chop of from the bottom
    backend_cutoff = config.get('backend_cutoff', 0)
    mil_mode = config.get('mil_mode', 'embedding')

    attribute = config.get('prediction_target', 'Age')

    # whether to drop all patients that have no value for the main attribute
    drop_na_main_attribute_values = config.get('drop_na_main_attribute_values', True)

    if attribute not in problem_kind:
        raise ValueError(f'Unknown attribute {attribute}')

    # TRAINING ASPECTS
    batch_size = config.get('batch_size', 4)
    # more than two bags per batch are likely to overwhelm memory

    # whether to crop images to ImageNet size (i.e. 224 * 224)
    limit_image_size = config.get('limit_image_size', True)
    use_mask = config.get('use_mask', False)
    lr = config.get('lr', 0.001)
    # separate lr for the backend can be specified, defaults to normal LR
    backend_lr = config.get('backend_lr', lr)

    n_epochs = config.get('n_epochs', 20)

    in_channels = 1 if use_one_channel else 3
    config['in_channels'] = in_channels
    # hand over to backend
    backend_kwargs = {'pretrained': pretrained, 'feature_extraction': feature_extraction, 'in_channels': in_channels}

    use_pseudopatients = config.get('use_pseudopatients', False)

    # Multi-Head
    # loss weights for every attribute
    att_loss_weights = config.get('att_loss_weights', {})
    # get all attributes
    additional_atts = list(att_loss_weights.keys())
    # remove the base attribute here, as it's not additional
    if attribute in additional_atts:
        additional_atts.remove(attribute)

    # DA
    # warn user if lambda weight specified but no DA
    if 'lambda_weight' in config and not 'target_train' in config:
        warnings.warn('Specified weight for DA but did not provide target set!')
    lambda_weight = config.get('lambda_weight', 0.5)

    layers_to_compute_da_on = config.get('layers_to_compute_da_on', [1])

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

    # get all attribute specs for the specified problem
    attribute_specs = []
    # yields all possible attributes to predict
    att_spec_dict = make_att_specs()
    all_atts = [attribute] + additional_atts
    for attribute_name in all_atts:
        attribute_specs.append(att_spec_dict[attribute_name])

    # the classes we predict
    train_classes = att_spec_dict[attribute].legal_values
    # whether or not to drop all patients with na values depends on single vs multi-head
    multihead = bool(additional_atts)
    print(f'Performing multi-head classification?: {multihead}')
    # listen to user specified input in case it is multihead, else, always drop na values
    dropna_values = drop_na_main_attribute_values if multihead else True

    # filter_attribute = 'Class_sample' if attribute == 'Class' else None

    # this is always needed
    source_train = config.get('source_train')
    target_train = config.get('target_train', None)
    val = config.get('val')

    # 'target_train': SetSpec('Philips_iU22', 'umc', 'train', umc_data_path)
    desired_set_specs = {'source_train': source_train, 'target_train': target_train,
                         'val': val}

    # yields a mapping from names to set_specs
    set_spec_dict = make_set_specs(umc_data_path, umc_img_root, jhu_data_path, jhu_img_root)

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

    muscles_to_use = None
    use_most_frequent_muscles = config.get('muscle_subset', False)
    if use_most_frequent_muscles:
        muscles_to_use = ['Biceps brachii', 'Tibialis anterior', 'Gastrocnemius medial head', 'Flexor carpi radialis',
                          'Vastus lateralis']

    dataset_storage = {}
    train_transform_params = None
    for set_name, set_spec_list in desired_set_specs.items():
        datasets = []
        # bag_loaders = []
        for set_spec in set_spec_list:
            print(set_spec)
            # always drop na values for the validation or test set
            dropna_values_set = dropna_values if ('train' in set_name) else True
            # pass the classes in to ensure that only those are present in all the sets
            patients = get_data_for_spec(set_spec, loader_type='bag', attribute_to_filter=attribute,
                                         legal_attribute_values=train_classes,
                                         muscles_to_use=muscles_to_use,
                                         dropna_values=dropna_values_set)

            # patients = patients[0:40]
            print(f'Loaded {len(patients)} elements.')

            img_path = set_spec.img_root_path

            use_pseudopatient_locally = (set_name != 'val') & use_pseudopatients
            # loader = make_bag_loader(patients, img_path, use_one_channel, normalizer_name,
            #                          attribute_specs, batch_size, set_spec.device, limit_image_size=limit_image_size,
            #                          use_pseudopatients=use_pseudopatient_locally,
            #                          pin_memory=use_cuda, return_attribute_dict=True)
            # bag_loaders.append(loader)

            transform_params = {'resize_option_name' : set_spec.device, 'normalizer_name': normalizer_name,
                                'limit_image_size': limit_image_size}

            # store the transform params used for training
            if set_name == 'source_train':
                train_transform_params = transform_params

            transform = make_basic_transform(**transform_params)

            ds = make_bag_dataset(patients, img_path, use_one_channel=use_one_channel,
                                     attribute_specs=attribute_specs, transform=transform,
                                     use_pseudopatients=use_pseudopatient_locally,
                                     return_attribute_dict=True, use_mask=use_mask)

            datasets.append(ds)

        dataset_storage[set_name] = datasets

    perform_da = ('source_train' in dataset_storage) and ('target_train' in dataset_storage)

    if perform_da:
        # make a combined dataset for both
        ds = ConcatDataset(*[dataset_storage['source_train'][0], dataset_storage['target_train'][0]])
        dataset_storage.pop('target_train')
        dataset_storage.pop('source_train')

        dataset_storage['train'] = ds

    else:
        dataset_storage['train'] = dataset_storage['source_train']
        dataset_storage.pop('source_train')

    set_loaders = {}
    # create data loaders for all the sets
    for set_name, dataset_list in dataset_storage.items():
        if not isinstance(dataset_list, list):
            dataset_list = [dataset_list]
        bag_loaders = []
        for ds in dataset_list:
            loader = wrap_in_bag_loader(ds, batch_size, pin_memory=use_cuda, return_attribute_dict=True)
            bag_loaders.append(loader)
        set_loaders[set_name] = bag_loaders

    model_param_dict = {'att_specs' : attribute_specs, 'backend': backend, 'mil_pooling' : mil_pooling,
                      'pooling_kwargs': pooling_kwargs, 'mode':mil_mode, 'fc_hidden_layers':fc_hidden_layers,
                      'fc_use_bn': fc_use_bn,
                      'backend_cutoff':backend_cutoff,
                      'backend_kwargs':backend_kwargs}


    # create the desired model
    model = MultiInputNet(**model_param_dict)

    n_params_backend = pytorch_count_params(model.backend)
    n_params_classifier = pytorch_count_params(model.heads[attribute])
    n_params_pooling = pytorch_count_params(model.mil_pooling)
    config['n_params_classifier'] = n_params_classifier
    config['n_params_pooling'] = n_params_pooling

    # can specify different learning rates for each component
    optimizer = optim.Adam([{'params': model.backend.parameters(), 'lr': backend_lr}], lr)

    config['n_params_backend'] = n_params_backend

    device = torch.device("cuda:0" if use_cuda else "cpu")
    # needs to be manually enforced to work on the cluster
    model.to(device)

    # this custom transform allows attaching metrics directly to the trainer
    # as y and y_pred can be read out from the output dict
    def custom_output_transform(x, y, y_pred, loss):
        base_dict = {
            "y": y,
            "y_pred": y_pred['preds'],
            "loss": loss.item()
        }
        if 'atts' in y_pred:
            base_dict['atts'] = y_pred['atts']
        if 'coral_losses' in y_pred:
            base_dict['coral_losses'] = y_pred['coral_losses']
        return base_dict

    # allow adding a weight to reweight binary labels --> trade in precision and recall
    if 'class_pos_weight' in config:
        # kwargs for the loss function
        # wrap in list
        loss_weight = [config['class_pos_weight']]
        # make kwargs
        loss_kwargs = {'pos_weight': torch.FloatTensor(loss_weight).to(device=device)}
        loss_kwargs_mapping = {'Class': loss_kwargs}
    else:
        loss_kwargs_mapping = {}

    metrics = obtain_metrics(attribute_specs=attribute_specs, extract_multiple_atts=True,
                             loss_kwargs_mapping=loss_kwargs_mapping)

    # logging attention distributions
    log_attention = mil_pooling == 'attention'
    if log_attention:
        metrics_to_add = {'mean_att': Average(output_transform=lambda output: output['atts']),
                          'var_att': Variance(output_transform=lambda output: output['atts']),
                          'min_att': Minimum(output_transform=lambda output: output['atts']),
                          'max_att': Maximum(output_transform=lambda output: output['atts'])}
        metrics = {**metrics, **metrics_to_add}

    if perform_da:
        coral_metrics = {}
        # add a custom metric for passing through the coral losses
        for att_spec in attribute_specs:
            coral_metrics['mean_coral_' + att_spec.name] = Average(output_transform=lambda output: output['coral_losses'][att_spec.name])

        train_metrics = {**metrics, **coral_metrics}

        trainer = create_da_trainer(model, optimizer, attribute_specs,
                                    att_loss_weights=att_loss_weights, lambda_weight=lambda_weight,
                                    layers_to_compute_da_on=layers_to_compute_da_on,
                                    loss_kwargs_mapping= loss_kwargs_mapping,
                                    device=device, output_transform=custom_output_transform,
                                    deterministic=True)
    else:

        train_metrics = metrics
        trainer = create_bag_attention_trainer(model, optimizer, attribute_specs,
                                               att_loss_weights=att_loss_weights, loss_kwargs_mapping=loss_kwargs_mapping,
                                               device=device, output_transform=custom_output_transform,
                                               deterministic=True)

    # attach metrics to the trainer
    for set_spec, metric in train_metrics.items():
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
        base_dict = {
            "y": y,
            "y_pred": y_pred['preds'],
        }
        if 'atts' in y_pred:
            base_dict['atts'] = y_pred['atts']
        return base_dict

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

    checkpoint_base_path = os.path.join(mnt_path, 'klaus/muscle-ultrasound/checkpoints')

    # fallback to naming based on experiment config
    if 'checkpoint_dir' not in config:
        try:
            exp_id = npt_logger.get_experiment()._id
            checkpoint_dir = os.path.join(checkpoint_base_path, project_name, exp_id)
            os.makedirs(checkpoint_dir, exist_ok=True)
        except:
            checkpoint_dir = checkpoint_base_path
    else:
        checkpoint_dir = config.get('checkpoint_dir')

    print(f'Using checkpoint: {checkpoint_dir}')

    checkpointer = ModelCheckpoint(checkpoint_dir, 'pref', create_dir=True, require_empty=False, n_saved=None,
                                   global_step_transform= global_step_from_engine(trainer))
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), checkpointer,
                              {'model_state_dict': model, 'model_param_dict': StateDictWrapper(model_param_dict),
                               'transform_dict': StateDictWrapper(train_transform_params), 'config': StateDictWrapper(config)})

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
    train_loader = set_loaders['train'][0]
    trainer.run(train_loader, max_epochs=n_epochs)

    npt_logger.close()

if __name__ == '__main__':
    # TODO read out from argparse
    bag_config = {'problem_type': 'bag', 'prediction_target': 'Class', 'backend_mode': 'finetune',
                  'backend': 'resnet-18', 'mil_pooling': 'mean', 'attention_mode': 'sigmoid',
                  'mil_mode': 'embedding', 'batch_size': 4, 'lr': 0.0269311, 'n_epochs': 5,
                  'use_pseudopatients': False, 'fc_hidden_layers': 2, 'fc_use_bn': True,
                  'backend_cutoff': 1}

    only_philips = {'source_train': 'Philips_iU22_train',
                         'val': 'Philips_iU22_val'}

    only_esaote = {'source_train': 'ESAOTE_6100_train',
                         'val': 'ESAOTE_6100_val'}

    da = {'source_train': 'ESAOTE_6100_train', 'target_train': 'Philips_iU22_train',
                         'val': ['ESAOTE_6100_val', 'Philips_iU22_val']}
    config = {**bag_config, **da}

    train_multi_input(config)
