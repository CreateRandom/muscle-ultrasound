from functools import partial

import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint
from torchvision.datasets import DatasetFolder
from torchvision.transforms import transforms

from loading.img_utils import load_dicom
from models.premade import make_resnet_18

def make_umc_set(base_path, load_mask=False, use_one_channel=False, normalize=True):
    """

    :param base_path: A path that contains a folder for each class, which in turn contains dicom files
    and optionally mat files with ROIs to be used as mask
    :param load_mask: Whether mat files should be loaded and used as masks
    :return: A torch.utils.data.Dataset
    """
    # for now, use a standard class with a dicom loader, might want to come up with a custom format
    loader = partial(load_dicom, load_mask=load_mask, use_one_channel=use_one_channel)

    # image size to rescale to
    r = transforms.Resize((224, 224))

    t_list = [r, transforms.ToTensor()]

    # we can only normalize if we use all three channels
    if normalize and not use_one_channel:
        # necessary to leverage the pre-trained models properly
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        t_list.append(normalize)

    c = transforms.Compose(t_list)

    return DatasetFolder(root=base_path, loader=loader, extensions=('dcm',), transform=c)

torch.manual_seed(42)
base_path = '/mnt/chansey/klaus/Gastrocnemius'

use_one_channel = False
pretrained = True
feature_extraction = True

ds = make_umc_set(base_path, use_one_channel=use_one_channel, normalize=False)

train_size = int(0.8 * len(ds))
val_size = int(0.1 * len(ds))
test_size = len(ds) - train_size - val_size
# todo: need a more principled way to split data
train_dataset, val_dataset, test_dataset = random_split(ds, [train_size, val_size, test_size])


# for i in range(3):
#     a, y = ds[i]
#     plt.imshow(a)
#     plt.show()

batch_size = 4
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True,num_workers=4)

in_channels = 1 if use_one_channel else 3
model = make_resnet_18(num_classes=2, pretrained=pretrained, in_channels=in_channels,feature_extraction=feature_extraction)

# todo investigate / tune
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=10e-2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
evaluator = create_supervised_evaluator(model,
                                        metrics={
                                            'accuracy': Accuracy(),
                                            'p': Precision(),
                                            'r': Recall(),
                                            'nll': Loss(criterion)
                                            }, device=device)

pbar = ProgressBar()
pbar.attach(trainer)


checkpoint_dir = 'checkpoints'

checkpointer = ModelCheckpoint(checkpoint_dir, 'pref', n_saved=3, create_dir=True, require_empty=False)
trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1),checkpointer, {'mymodel' : model})

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics

    p = metrics['p'].to('cpu').numpy()[1]
    r = metrics['r'].to('cpu').numpy()[1]

    print("Training Results - Epoch: {} Avg accuracy: {:.2f} Avg precision: {:.2f} Avg recall: {:.2f} Avg loss: {:.2f}"
          .format(trainer.state.epoch, metrics['accuracy'], p, r , metrics['nll']))

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics

    p = metrics['p'].to('cpu').numpy()[1]
    r = metrics['r'].to('cpu').numpy()[1]
    print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg precision: {:.2f} Avg recall: {:.2f} Avg loss: {:.2f}"
          .format(trainer.state.epoch, metrics['accuracy'], p,r ,metrics['nll']))

trainer.run(train_loader, max_epochs=5)

