import torch

from torch import nn

class MultiInputNet(nn.Module):

    def __init__(self, num_classes=2):
        super(MultiInputNet, self).__init__()

        self.channels = 3
        self.features = nn.Sequential(
            nn.Conv2d(1, self.channels, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(self.channels, 3 * self.channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(3 * self.channels, 6 * self.channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(6 * self.channels, 4 * self.channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * self.channels, 4 * self.channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(5 * (4 * self.channels) * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):#
        # batch * muscle * resolution
        # split into batches of one channel / muscle each
        channels = torch.split(x, split_size_or_sections=1, dim=1)
        # apply batch-wise per channel
        feature_reps = [self.features(x) for x in channels]
        # pool per channel
        pooled = [self.avgpool(x) for x in feature_reps]
        # merge all into batch * n_channel * ...
        merged = torch.stack(pooled, dim=1)
        # flatten and classify
        x = torch.flatten(merged, 1)
        x = self.classifier(x)
        return x
