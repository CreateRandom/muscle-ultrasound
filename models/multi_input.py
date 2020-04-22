import torch

from torch import nn
import torch.nn.functional as F

from models.mil_pooling import MaxMIL, AttentionMIL, AverageMIL, GatedAttentionMIL
from models.premade import make_resnet_18, make_alexnet


class MultiInputAlexNet(nn.Module):

    def __init__(self, num_classes=2, inner_channels=64):
        super(MultiInputAlexNet, self).__init__()

        self.channels = inner_channels

        self.features = nn.Sequential(
            # this takes in one channel at a time
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

    def forward(self, x):  #
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


class DefaultBackend(nn.Module):
    def __init__(self, n_out):
        super(DefaultBackend, self).__init__()
        self.feature_extractor_1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))

        self.feature_extractor_2 = nn.Linear(50 * 4 * 4, n_out)

    def forward(self, x):
        x = self.feature_extractor_1(x)
        x = x.view(-1, 50 * 4 * 4)
        x = self.feature_extractor_2(x)
        return x


def make_default_backend(use_embedding, backend_kwargs):
    out_dim = 1
    if use_embedding:
        out_dim = 500

    return DefaultBackend(out_dim)

def make_resnet_backend(use_embedding, backend_kwargs):
    out_dim = 1
    backend = make_resnet_18(num_classes=1, **backend_kwargs)
    if use_embedding:
        backend.fc = nn.Identity()
        out_dim = 512
    return backend, out_dim

def make_alex_backend(use_embedding, backend_kwargs):
    out_dim = 1
    backend = make_alexnet(num_classes=1, **backend_kwargs)
    if use_embedding:
        backend.classifier[6] = nn.Identity()
        out_dim = 4096
    return backend, out_dim


backend_funcs = {'resnet-18': make_resnet_backend, 'default': make_default_backend, 'alexnet': make_alex_backend}

# for now, we will always assume binary classification, maybe later see how to expand this
class MultiInputNet(nn.Module):
    def __init__(self, backend='alexnet', mil_pooling='attention', classifier='fc', mode='embedding', backend_kwargs=None):
        super(MultiInputNet, self).__init__()
        self.backend_type = backend
        self.mode = mode
        self.backend_kwargs = {} if not backend_kwargs else backend_kwargs

        self.mil_pooling_type = mil_pooling
        self.classifier_type = classifier
        # TODO allow for multiple backends
        self.backend, self.backend_out_dim = self.make_backend(backend, mode, self.backend_kwargs)
        # TODO allow for multiple pooling functions
        self.mil_pooling = self.make_mil_pooling(mil_pooling, self.backend_out_dim)
        # TODO allow for multiple classifiers
        self.classifier = self.make_classifier(classifier, self.backend_out_dim)

    def __str__(self):
        return f'MultiInputNet with {self.backend_type} ' \
               f'backend, {self.mil_pooling_type} mil and {self.classifier_type} classifier'

    def forward(self, x):
        # input here has batch (always 1) * bag * channels * px * px
        x = x.squeeze(0)
        x = self.backend(x)
        x = self.mil_pooling(x)
        x = self.classifier(x)
        return x

    @staticmethod
    def make_backend(backend_name, mode, backend_kwargs):

        # think about the best way to do this
        backend_func = backend_funcs[backend_name]
        use_embedding = (mode == 'embedding')
        backend, out_dim = backend_func(use_embedding, backend_kwargs)

        # TODO come up with a more general way to do this
        if mode == 'embedding':
            backend = nn.Sequential(backend, nn.ReLU())
        elif mode == 'instance':
            backend = nn.Sequential(backend, nn.Sigmoid())
        else:
            raise ValueError(f'Unknown mode: {mode}')
        return backend, out_dim

    @staticmethod
    def make_mil_pooling(type, in_dim):
        if type == 'max':
            return MaxMIL()
        elif type == 'mean':
            return AverageMIL()
        elif type == 'attention':
            return AttentionMIL(in_dim, 128, 1)
        elif type == 'gated_attention':
            return GatedAttentionMIL(in_dim, 128, 1)
        else:
            raise ValueError(f'Invalid MIL type: {type}')

    @staticmethod
    def make_classifier(type, in_dim):
        # TODO dynamically figure out the dimensions here
        classifier = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.Sigmoid()
        )
        return classifier

def BernoulliLoss(output, target):
    # apply the sigmoid here
    # Y_sig = nn.Sigmoid()(output)
    # Y_pos = Y_sig[:,1]
    Y_prob = torch.clamp(output, min=1e-5, max=1. - 1e-5)
    Y = target
    neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

    return neg_log_likelihood.squeeze()