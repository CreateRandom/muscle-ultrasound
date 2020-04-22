from torch.utils.data import DataLoader

from loading.loaders import make_myositis_loaders, make_umc_loaders
from loading.mnist_bags import MnistBags
from models.multi_input import MultiInputNet, BernoulliLoss

import torch
from torch import nn, optim

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

from utils.ignite_utils import ValueCount

y_epoch = []
ypred_epoch = []
losses = []

def train_model(config):
    # BACKEND aspects
    pretrained = config.get('pretrained', True)
    feature_extraction = config.get('feature_extraction', True)
    use_one_channel = config.get('use_one_channel', False)
    # normalize when necessary
    normalize = not use_one_channel and pretrained

    backend = config.get('backend','resnet-18')
    mil_pooling = config.get('mil_pooling','attention')
    classifier = config.get('classifier','fc')
    mil_mode = config.get('mil_mode','embedding')

    # TRAINING ASPECTS
    batch_size = config.get('batch_size', 1)
    lr = config.get('lr', 0.001)
    n_epochs = config.get('n_epochs', 20)

    in_channels = 1 if use_one_channel else 3
    # hand over to backend
    backend_kwargs = {'pretrained': pretrained, 'feature_extraction': feature_extraction, 'in_channels': in_channels}

    mode = 'umc'

    if mode == 'myositis':
        train_path = '/home/klux/Thesis_2/data/myositis/train.csv'
        val_path = '/home/klux/Thesis_2/data/myositis/val.csv'
        img_folder = '/home/klux/Thesis_2/data/myositis/processed_imgs'
        attribute = 'Diagnosis_bin'
        train_loader, val_loader = make_myositis_loaders(train_path, val_path, img_folder, use_one_channel, normalize, attribute, batch_size)
    # for purposes of comparison
    elif mode == 'mnist_bags':
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
    elif mode == 'umc':
        train_path = '/mnt/chansey/klaus/all_patients/full_format_image_info_train.csv'
        val_path = '/mnt/chansey/klaus/all_patients/full_format_image_info_val.csv'
        img_folder = '/mnt/chansey/klaus/all_patients/'
        attribute = 'Class'
        train_loader, val_loader = make_umc_loaders(train_path, val_path, img_folder, use_one_channel, normalize,
                                                         attribute, batch_size)
    else:
        raise ValueError(f'Invalid mode : {mode}')

    def binarize_sigmoid(y_pred):
        return torch.ge(y_pred,0.5).int()

    # if we have an FC as the last layer
    def binarize_predictions(y_pred):
        sm = nn.Softmax(dim=1)(y_pred)
        _, i = torch.max(sm, dim=1)
        return i

    def binarize_output(output):
        y_pred, y = output
        y_pred = binarize_sigmoid(y_pred)
        return y_pred, y

    model = MultiInputNet(backend=backend, mil_pooling=mil_pooling,
                          classifier=classifier, mode=mil_mode, backend_kwargs=backend_kwargs)

    # todo investigate / tune
    criterion = BernoulliLoss
    optimizer = optim.SGD(model.parameters(), lr)
    # optimizer = optim.Adam(model.parameters(), lr)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # called on every iteration
    def evaluate_training_items(x, y, y_pred, loss):
        preds = binarize_sigmoid(y_pred)
        y_epoch.extend(y.cpu().numpy().tolist())
        ypred_epoch.extend(preds.cpu().numpy().tolist())
        losses.append(loss.item())
        return loss.item()

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device,
                                        output_transform=evaluate_training_items)

    evaluator = create_supervised_evaluator(model,
                                            metrics={
                                                'accuracy': Accuracy(output_transform=binarize_output),
                                                'p': Precision(output_transform=binarize_output),
                                                'r': Recall(output_transform=binarize_output),
                                                # TODO fixme
                                              #  'counts': ValueCount(output_transform=binarize_output),
                                                'nll': Loss(criterion)
                                            }, device=device)

    pbar = ProgressBar()
    pbar.attach(trainer)

    checkpoint_dir = 'checkpoints'

    checkpointer = ModelCheckpoint(checkpoint_dir, 'pref', n_saved=3, create_dir=True, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), checkpointer, {'mymodel': model})

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        # evaluator.run(train_loader)
        # metrics = evaluator.state.metrics

        global losses, y_epoch, ypred_epoch
        loss = np.mean(losses)
        acc = accuracy_score(y_epoch, ypred_epoch)
        p = precision_score(y_epoch, ypred_epoch)
        r = recall_score(y_epoch, ypred_epoch)
        print(
            "Training Results - Epoch: {} Avg accuracy: {:.2f} Avg precision: {:.2f} Avg recall: {:.2f} Avg loss: {:.2f}"
                .format(trainer.state.epoch, acc, p, r, loss))
        # print(y_epoch)
        # print(ypred_epoch)
        # stores training preds
        y_epoch = []
        ypred_epoch = []
        losses = []

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} "
            "Avg precision: {:.2f} Avg recall: {:.2f} Avg loss: {:.2f}"
            .format(trainer.state.epoch, metrics['accuracy'], metrics['p'], metrics['r'], metrics['nll']))#, metrics['counts']))

    #  tune.track.log(iter=evaluator.state.epoch, mean_accuracy=metrics['accuracy'])

    trainer.run(train_loader, max_epochs=n_epochs)


train_model({})

# ray.init(num_gpus=1)
#
# analysis = tune.run(
#     train_model, config={"lr": tune.grid_search([0.001, 0.01])},
#     resources_per_trial={"gpu": 1}
# )
