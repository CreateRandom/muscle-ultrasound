import functools

import torch

from torch import nn
import numpy as np
from models.mil_pooling import MaxMIL, AttentionMIL, AverageMIL, GatedAttentionMIL
from models.premade import make_resnet_18, make_alexnet
from utils.metric_utils import _binarize_softmax, _binarize_sigmoid


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


def make_default_backend(use_embedding, cutoff, backend_kwargs):
    out_dim = 1
    if use_embedding:
        out_dim = 500

    return DefaultBackend(out_dim), out_dim

def make_resnet_backend(use_embedding, cutoff, backend_kwargs):
    if use_embedding:
        # get the full resnet (including the readout layer)
        backend = make_resnet_18(**backend_kwargs)
        out_dim = 1000
        if cutoff > 0:
            backend.fc = nn.Identity()
            out_dim = 512
        if cutoff > 1:
            # can set various layers to identity
            backend.layer4 = nn.Identity()
            out_dim = 256
        if cutoff > 2:
            backend.layer3 = nn.Identity()
            out_dim = 128
        if cutoff > 3:
            backend.layer2 = nn.Identity()
            out_dim = 64
        if cutoff > 4:
            backend.layer1 = nn.Identity()
            out_dim = 64
    else:
        # make a resnet with the last fc layer for scoring
        backend = make_resnet_18(num_classes=1, **backend_kwargs)
        out_dim = 1
    return backend, out_dim
# todo add cutoff
def make_alex_backend(use_embedding, cutoff, backend_kwargs):
    out_dim = 1
    backend = make_alexnet(num_classes=1, **backend_kwargs)
    if use_embedding:
        backend.classifier[6] = nn.Identity()
        out_dim = 4096
    return backend, out_dim

class MultiInputBaseline(nn.Module):
    def __init__(self, image_level_classifier, label_type):
        super(MultiInputBaseline, self).__init__()
        self.image_level_classifier = image_level_classifier
        # use this only for predictions
        # TODO make sure this doesn't mess with the gradient
        # self.image_level_classifier.eval()

        self.mode = label_type
        if label_type == 'regression':
            self.out_transform = nn.Identity()
            self.aggregate = torch.mean
        elif label_type == 'binary':
            self.out_transform = _binarize_sigmoid
            self.aggregate = lambda x: torch.mode(x,dim=0)[0]
        elif label_type == 'multi':
            self.out_transform = _binarize_softmax
            self.aggregate = lambda x: torch.mode(x,dim=0)[0]
        else:
            raise ValueError(f'Unrecognized mode {label_type}')

    def forward(self, x):
        # img_rep: n_total_images * channels * height * width
        # n_images_per_bag: how to allocate images to bags
        img_rep, n_images_per_bag = x
        n_images_per_bag = tuple(n_images_per_bag.cpu().numpy())

        # get the predictions for all images at the same time
        preds = self.image_level_classifier(img_rep)

        # split by patient
        pwise_preds = torch.split(preds, n_images_per_bag)
        final_preds = []
        for preds in pwise_preds:
            # perform the output transform to get labels
            transformed_preds = self.out_transform(preds)
            final_pred = self.aggregate(transformed_preds)
            final_preds.append(final_pred)

        final_preds = torch.stack(final_preds)
        return final_preds

backend_funcs = {'resnet-18': make_resnet_backend, 'default': make_default_backend, 'alexnet': make_alex_backend}
cnn_constructors = {'resnet-18': make_resnet_18, 'alexnet': make_alexnet}
# for now, we will always assume binary classification, maybe later see how to expand this
class MultiInputNet(nn.Module):
    def __init__(self, att_specs, backend='alexnet', mil_pooling='attention', mode='embedding',
                 fc_hidden_layers=0, fc_use_bn=True, backend_cutoff=0,
                 backend_kwargs=None, pooling_kwargs=None):
        super(MultiInputNet, self).__init__()
        self.backend_type = backend
        self.mode = mode
        default_pooling = {'mode': 'identity', 'D': 128}


        self.pooling_kwargs = default_pooling if not pooling_kwargs else pooling_kwargs
        self.backend_kwargs = {} if not backend_kwargs else backend_kwargs
        self.mil_pooling_type = mil_pooling
        # allows for multiple backends
        self.backend, self.backend_out_dim = self.make_backend(backend, mode, backend_cutoff, self.backend_kwargs)
        # allows for multiple pooling functions
        self.mil_pooling = self.make_mil_pooling(mil_pooling, self.backend_out_dim, self.pooling_kwargs)
        # allow for variation
        # hidden_dims, in_dim, out_dim, activation='relu', bn=True
        # add a hidden dimension with half the size of the backend output
        self.hidden_dims = []
        dim = self.backend_out_dim
        for ind in range(fc_hidden_layers):
            dim = int(np.ceil(dim / 2))
            if dim > 1:
                self.hidden_dims.append(dim)
        print(f'Using hidden dims {self.hidden_dims}')
   #     self.classifier = self.make_classifier(hidden_dims=self.hidden_dims, in_dim=self.backend_out_dim,
   #                                            out_dim=self.out_dim, activation='leaky_relu',bn=fc_use_bn)

        self.heads = {}
        for att_spec in att_specs:
            if att_spec.target_type == 'regression' or att_spec.target_type == 'binary':
                out_dim = 1
            else:
                out_dim = len(att_spec.legal_values)
            classifier = self.make_classifier(hidden_dims=self.hidden_dims, in_dim=self.backend_out_dim,
                                               out_dim=out_dim, activation='leaky_relu',bn=fc_use_bn)
            self.add_module(att_spec.name  + '_head', classifier)
            self.heads[att_spec.name] = classifier


    def __str__(self):
        return f'MultiInputNet with {self.backend_type} ' \
               f'backend, {self.mil_pooling_type} mil and {self.hidden_dims} classifier'

    # for the case where the batch size is one
    def forward_single(self, x):
        # input here has batch (always 1) * bag * channels * px * px
        x = x.squeeze(0)
        x = self.backend(x)
        x = self.mil_pooling(x)
        x = self.classifier(x)
        # 1 * 1
        return x

    def forward(self, x):
        # img_rep: n_total_images * channels * height * width
        # n_images_per_bag: how to allocate images to bags
        img_rep, n_images_per_bag = x
        n_images_per_bag = tuple(n_images_per_bag.cpu().numpy())

        # patients (batch size) * n_images_per_patient * channels * height * width
        # n_patients = x.shape[-5]
        # n_images_per_patient = x.shape[-4]
        # all_images_flat = x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])

        # get the backend output for all images at the same time
        all_mats_flat = self.backend(img_rep)
        # split by patient
        pwise_images = torch.split(all_mats_flat, n_images_per_bag)

        # reshape back to original format
        # mil_inputs = all_mats_flat.reshape(n_patients, n_images_per_patient, self.backend_out_dim)
        pooled = []
        attention_outputs = []
        # for elem in pwise_images:
        #     pooled.append(self.classifier(self.mil_pooling(elem)).squeeze())
        # pooled = torch.stack(pooled)

        for elem in pwise_images:
            p, att = self.mil_pooling(elem)
            pooled.append(p.squeeze())
            if att is not None:
                attention_outputs.append(att.clone().detach())
        pooled = torch.stack(pooled)

        # ensure batch first format
        if pooled.ndimension() == 1:
             pooled = pooled.unsqueeze(0)

       # pooled = self.classifier(pooled)
        head_outputs = {}
        for name, head in self.heads.items():
            output = head(pooled)
            if output.ndimension() == 1:
                output = output.unsqueeze(1)
            head_outputs[name] = output

        # # ensure batch first format
        # if pooled.ndimension() == 1:
        #     pooled = pooled.unsqueeze(1)

        return head_outputs, attention_outputs

    @staticmethod
    def make_backend(backend_name, mode, backend_cutoff, backend_kwargs):

        # think about the best way to do this
        backend_func = backend_funcs[backend_name]
        use_embedding = (mode == 'embedding')
        backend, out_dim = backend_func(use_embedding, backend_cutoff, backend_kwargs)

        if mode == 'embedding':
            backend = nn.Sequential(backend, nn.ReLU())
        elif mode == 'instance':
            backend = nn.Sequential(backend, nn.Sigmoid())
        else:
            raise ValueError(f'Unknown mode: {mode}')
        return backend, out_dim

    @staticmethod
    def make_mil_pooling(type, in_dim, pooling_kwargs=None):
        if type == 'max':
            return MaxMIL()
        elif type == 'mean':
            return AverageMIL()
        elif type == 'attention':
            return AttentionMIL(L=in_dim, K=1, **pooling_kwargs)
        elif type == 'gated_attention':
            return GatedAttentionMIL(L=in_dim, K=1, **pooling_kwargs)
        else:
            raise ValueError(f'Invalid MIL type: {type}')

    @staticmethod
    def make_classifier(hidden_dims, in_dim, out_dim, activation='relu', bn=False):

        mlp = []

        if activation == 'relu':
            act_fn = functools.partial(nn.ReLU, inplace=True)
        elif activation == 'leaky_relu':
            act_fn = functools.partial(nn.LeakyReLU, inplace=True)
        else:
            raise NotImplementedError
        for hidden_dim in hidden_dims:
            mlp += [nn.Linear(in_dim, hidden_dim)]
            if bn:
                # input needs to be batch * in * (channels)
                mlp += [nn.BatchNorm1d(hidden_dim)]
            mlp += [act_fn()]
            in_dim = hidden_dim

        # add the readout layer at the very end
        mlp += [nn.Linear(in_dim, out_dim)]
        # connect into sequential
        classifier = nn.Sequential(*mlp)

        return classifier

def BernoulliLoss(output, target):
    # apply the sigmoid here
    # Y_sig = nn.Sigmoid()(output)
    # Y_pos = Y_sig[:,1]
    # added
  #  output = output.squeeze()

    Y_prob = torch.clamp(output, min=1e-5, max=1. - 1e-5)
    Y = target
    neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
    return torch.sum(neg_log_likelihood)
   # return neg_log_likelihood.squeeze()