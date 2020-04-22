import ray
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from loading.mnist_bags import MnistBags
from loading.datasets import PatientChannelDataset, PatientBagDataset
from models.multi_input import MultiInputAlexNet, MultiInputNet, BernoulliLoss
from models.premade import make_resnet_18, make_alexnet
from ray import tune

import torch
from torch import nn, optim

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint
from imgaug.augmenters import RandAugment
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

def make_patient_channel_loader(csv_path, root_folder, attribute, transform, batch_size, is_val, muscles_to_use):
    ds = PatientChannelDataset(patient_muscle_csv_chart=csv_path, root_dir=root_folder,
                               attribute=attribute, transform=transform, is_val=is_val, muscles_to_use=muscles_to_use)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader

class AugmentWrapper(object):
    def __init__(self, augment):
        self.augment = augment

    def __call__(self, img):
        return self.augment.augment_image(np.array(img))


def make_transform(use_augment):
    t_list= []
    # image size to rescale to
    r = transforms.Resize((224, 224))
    t_list.append(r)

    # data augmentation
    if use_augment:
        aug = AugmentWrapper(RandAugment())
        t_list.append(aug)

    t_list.append(transforms.ToTensor())

    return transforms.Compose(t_list)


y_epoch = []
ypred_epoch = []
losses = []

img_folder = '/home/klux/Thesis_2/data/myositis/processed_imgs'


def train_model(config):
    # BACKEND aspects
    pretrained = config.get('pretrained', False)
    feature_extraction = config.get('feature_extraction', False)
    batch_size = config.get('batch_size', 16)
    lr = config.get('lr', 10e-2)
    n_epochs = config.get('n_epochs', 5)

    attribute = 'Diagnosis_bin'
    muscles_to_use = ['D', 'B', 'FCR', 'R', 'G']

    train_transform = make_transform(use_augment=False)
    train_loader = make_patient_channel_loader('/home/klux/Thesis_2/data/myositis/train.csv', img_folder,
                                           attribute=attribute, transform=train_transform, batch_size=batch_size,
                                           is_val=False, muscles_to_use=muscles_to_use)

    val_transform = make_transform(use_augment=False)
    val_loader = make_patient_channel_loader('/home/klux/Thesis_2/data/myositis/val.csv', img_folder,
                                         attribute=attribute, transform=val_transform, batch_size=batch_size,
                                         is_val=True, muscles_to_use=muscles_to_use)

    def binarize_predictions(y_pred):
        sm = nn.Softmax(dim=1)(y_pred)
        _, i = torch.max(sm, dim=1)
        return i

    def binarize_output(output):
        y_pred, y = output
        return binarize_predictions(y_pred), y

    in_channels = len(muscles_to_use)

    model = MultiInputAlexNet()

    # todo investigate / tune
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # called on every iteration
    def evaluate_training_items(x, y, y_pred, loss):
        preds = binarize_predictions(y_pred)

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

        # stores training preds
        y_epoch = []
        ypred_epoch = []
        losses = []

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg precision: {:.2f} Avg recall: {:.2f} Avg loss: {:.2f}"
            .format(trainer.state.epoch, metrics['accuracy'], metrics['p'], metrics['r'], metrics['nll']))

    #  tune.track.log(iter=evaluator.state.epoch, mean_accuracy=metrics['accuracy'])

    trainer.run(train_loader, max_epochs=n_epochs)


train_model({})

# ray.init(num_gpus=1)
#
# analysis = tune.run(
#     train_model, config={"lr": tune.grid_search([0.001, 0.01])},
#     resources_per_trial={"gpu": 1}
# )
