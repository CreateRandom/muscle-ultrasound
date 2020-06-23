from sklearn.metrics import mean_absolute_error, accuracy_score
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss

from training_utils import problems, fix_seed
from loading.datasets import CustomLabelEncoder
from loading.loaders import make_bag_loader, \
    get_data_for_spec, make_image_loader, get_classes
from loading.loading_utils import make_set_specs
from models.multi_input import cnn_constructors, MultiInputBaseline
import os
import socket

import torch
from torch import optim

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall, MeanAbsoluteError, MetricsLambda, ConfusionMatrix
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import global_step_from_engine
from sklearn.dummy import DummyClassifier, DummyRegressor

# import logging
from ignite.contrib.handlers.neptune_logger import NeptuneLogger, OutputHandler, WeightsScalarHandler, \
    GradsScalarHandler

from utils.ignite_utils import PositiveShare, Variance, Average, binarize_softmax, binarize_sigmoid, \
    pytorch_count_params, create_image_to_bag_evaluator
from utils.tokens import NEPTUNE_API_TOKEN


def train_image_level(config):
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

    attribute = config.get('prediction_target', 'Age')

    if attribute not in problems:
        raise ValueError(f'Unknown attribute {attribute}')
    label_type = problems[attribute]
    is_classification = (label_type == 'multi' or (label_type == 'binary'))
    is_multi = (label_type == 'multi')
    # TRAINING ASPECTS
    batch_size = config.get('batch_size', 4)
    patient_batch_size = config.get('batch_size', 4)

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
    label_encoder = None
    train_classes = None
    muscles_to_use = None
    use_most_frequent_muscles = config.get('muscle_subset', False)
    if use_most_frequent_muscles:
        muscles_to_use = ['Biceps brachii', 'Tibialis anterior', 'Gastrocnemius medial head', 'Flexor carpi radialis',
                          'Vastus lateralis']

    for set_name, set_spec_list in desired_set_specs.items():

        bag_loaders = []
        image_loaders = []
        for set_spec in set_spec_list:
            print(set_spec)
            # pass the classes in to ensure that only those are present in all the sets
            patients = get_data_for_spec(set_spec, loader_type='bag', attribute=attribute,
                                         class_values=train_classes,
                                         muscles_to_use=muscles_to_use)
            print(f'Loaded {len(patients)} elements.')

            # if classification and this is the train set, we want to fit the label encoder on this
            if is_classification & (set_name == 'source_train'):
                train_classes = get_classes(patients, attribute)
                print(train_classes)
                label_encoder = CustomLabelEncoder(train_classes, one_hot_encode=False)
            print(get_classes(patients, attribute))
            img_path = set_spec.img_root_path
            # decide which type of loader we need here
            # make the bag loader
            loader = make_bag_loader(patients, img_path, use_one_channel, normalizer_name,
                                     attribute, patient_batch_size, set_spec.device, limit_image_size=limit_image_size,
                                     use_pseudopatients=False,
                                     label_encoder=label_encoder, pin_memory=use_cuda)
            bag_loaders.append(loader)

            # make the image loader
            images = get_data_for_spec(set_spec, loader_type='image', attribute=attribute,
                                       class_values=train_classes,
                                       muscles_to_use=muscles_to_use)
            image_loader = make_image_loader(images, img_path, use_one_channel, normalizer_name,
                                             attribute, batch_size, set_spec.device, limit_image_size=limit_image_size,
                                             label_encoder=label_encoder, pin_memory=use_cuda, is_multi=is_multi)
            image_loaders.append(image_loader)

        set_loaders['bag'][set_name] = bag_loaders
        set_loaders['image'][set_name] = image_loaders

    # output dimensionality of the network
    if train_classes and (len(train_classes) > 2):
        num_classes = len(train_classes)
    # regression or binary classification with sigmoid
    else:
        num_classes = 1

    # obtain baseline performance estimates

    print('Baseline scores')
    train_labels = set_loaders['bag']['source_train'][0].dataset.get_all_labels()
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

    for val_loader in set_loaders['bag']['val']:
        val_labels = val_loader.dataset.get_all_labels()
        val_preds = d.predict([0] * len(val_labels))
        score = scorer(val_labels, val_preds)
        print(score)

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
                                        device=device, output_transform=custom_output_transform,
                                        deterministic=True)

    # always log the loss
    metrics = {'loss': Loss(criterion, output_transform=lambda x: (x['y_pred'], x['y']))}

    # binary cases
    if is_classification & (num_classes == 1):
        metrics_to_add = {'accuracy': Accuracy(output_transform=binarize_sigmoid),
                          'p': Precision(output_transform=binarize_sigmoid),
                          'r': Recall(output_transform=binarize_sigmoid),
                          'pos_share': PositiveShare(output_transform=binarize_sigmoid),
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

    val_loaders_bag = set_loaders['bag']['val']
    val_evaluators_bag = []
    for set_spec, loader in zip(val_names, val_loaders_bag):
        # TODO make sure we use the correct metrics here

        metrics_to_add = {'accuracy': Accuracy(),
                          'p': Precision(),
                          'r': Recall(),
                          'pos_share': PositiveShare()
                          }
        #
        val_evaluator = create_supervised_evaluator(patient_eval, metrics=metrics_to_add, device=device,
                                                    output_transform=custom_output_transform_eval)

        # val_evaluator = create_image_to_bag_evaluator(model, metrics=metrics_to_add, device=device,
        #                                             output_transform=custom_output_transform_eval)
        # enable validation logging
        npt_logger.attach(val_evaluator,
                          log_handler=OutputHandler(tag=str(set_spec),
                                                    metric_names=list(metrics_to_add.keys()),
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
    config = {'prediction_target': 'Sex', 'backend_mode': 'finetune',
              'backend': 'resnet-18', 'batch_size': 32, 'lr': 0.0269311, 'n_epochs': 5}

    only_philips = {'source_train': 'Philips_iU22_train',
                    'val': 'Philips_iU22_val'}

    config = {**config, **only_philips}

    train_image_level(config)
