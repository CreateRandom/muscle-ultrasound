import torch
from torch import nn

import torchvision.models as models

def make_resnet_18(num_classes, pretrained=True, in_channels=3, feature_extraction=False):

    has_three_channels = (in_channels == 3)
    if pretrained and not has_three_channels:
        raise ValueError('Can only use pretrained model with three channels.')

    if not has_three_channels and feature_extraction:
        raise ValueError('Can only perform feature extraction for image with three channels.')

    # model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=pretrained)
    model = models.resnet18(pretrained=pretrained)

    if feature_extraction:
        # freeze all the layers
        for param in model.parameters():
            param.requires_grad = False

    # reset the initial conv layer if more or less than 3 channels are to be used
    if not has_three_channels:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)

    # reset final layer for desired classification
    model.fc = nn.Linear(512, num_classes)

    return model

def make_alexnet(num_classes, pretrained=True, in_channels=3, feature_extraction=False):

    has_three_channels = (in_channels == 3)
    if pretrained and not has_three_channels:
        raise ValueError('Can only use pretrained model with three channels.')

    if not has_three_channels and feature_extraction:
        raise ValueError('Can only perform feature extraction for image with three channels.')

   # model = torch.hub.load('pytorch/vision:v0.5.0', 'alexnet', pretrained=pretrained)
    model = models.alexnet(pretrained=pretrained)
    if feature_extraction:
        # freeze all the layers
        for param in model.parameters():
            param.requires_grad = False

    # reset the initial conv layer if more or less than 3 channels are to be used
    if not has_three_channels:
        model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2)

    # reset final layer for desired classification
    model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)

    return model