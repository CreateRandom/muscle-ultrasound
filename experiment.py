import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint

from loading.umc import make_umc_set
from models.premade import make_resnet_18

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

