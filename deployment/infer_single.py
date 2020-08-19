import json
import sys
import numpy as np
import itk
import os
import torch
from PIL import Image
from captum.attr._utils.visualization import _normalize_image_attr
from torch import nn
from captum.attr import Saliency
from captum.attr import visualization as viz

from deployment.load_multi_input import load_multi_input, load_transform_from_checkpoint, load_checkpoint
from utils.binarize_utils import _binarize_sigmoid


def expand_to_3d(img_array):
    return np.repeat(img_array[:, :, np.newaxis], 3, axis=2)


class ModelReadoutWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, img_rep, n_images_per_bag):
        x = tuple([img_rep, n_images_per_bag])
        model_return_dict = self.net(x)
        self.readout = model_return_dict
        return model_return_dict['head_preds']['Class']

    def format_readout(self):
        pred_dict = {}
        # TODO adjust behaviour for different heads based on type
        for name, tensor in self.readout['head_preds'].items():
            pred = _binarize_sigmoid(tensor).numpy()
            pred_dict[name] = pred.tolist()
        if 'attention_outputs' in self.readout:
            pred_dict['attention_outputs'] = self.readout['attention_outputs'][0].numpy().tolist()

        return pred_dict

if __name__ == '__main__':
    extensions = ('.mha', '.mhd')
    base_path = '../input'
    file_names = [fn for fn in os.listdir(base_path) if str.endswith(fn, extensions)]

    if not file_names:
        print("No files were found in the input", file=sys.stderr)
        sys.exit(-1)

    os.makedirs('../output/images',exist_ok=True)

    # load the model

    base_checkpoint_path = '/mnt/chansey/klaus/muscle-ultrasound/checkpoints/'
    project_name = 'createrandom/mus-experiments'
    experiment = 'MUS2-16'
    file_name = 'pref_checkpoint'
    epoch = 9

    file_name = file_name + '_' + str(epoch) + '.pt'

    # build path to the checkpoint
    checkpoint_dir = os.path.join(base_checkpoint_path, project_name, experiment, file_name)

    checkpoint = load_checkpoint(checkpoint_dir)
    config = checkpoint['config']
    model = load_multi_input(checkpoint)
    # the images already have to be resized by the exporter to allow storage in mha, so ignore the resize
    # specified in the output
    transform = load_transform_from_checkpoint(checkpoint, ignore_resize=True)
    model.eval()

    json_results = []
    # iterate over all images and store a result for each
    for fn in file_names:
        result = {}
        result['entity'] = fn
        result['error_messages'] = []
        impath = os.path.join(base_path, fn)
        image = itk.imread(impath)
        img_list = itk.array_from_image(image)
        # make sure that the dimensionality is correct, TODO parametrize
        if config['in_channels'] == 3:
            img_list = [expand_to_3d(elem) for elem in img_list]
            mode = 'RGB'
        else:
            mode = 'L'

        # load each image to PIL
        imgs = [Image.fromarray(img, mode=mode) for img in img_list]
        # apply the stored transform to the input
        imgs = [transform(img) for img in imgs]


        im_tensor = torch.stack(imgs)

        print(im_tensor.shape)
        n_img = torch.tensor([im_tensor.shape[0]], requires_grad=False)
        # wrap in a class that allows saliency readout
        wrapped_model = ModelReadoutWrapper(model)
        saliency = Saliency(wrapped_model)
        grads = saliency.attribute(inputs=im_tensor, target=0, additional_forward_args=n_img)
        # now the model has been invoked and the results are available
        results = wrapped_model.format_readout()
        norm_grads = []
        for i in range(len(grads)):
            # get the original image
            original_image = np.transpose((imgs[i].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
            # get the gradients for this image
            img_grad = grads[i, :, :, :]
            # shape must be h,w,c, move first dim to the end
            img_grad = img_grad.permute((1, 2, 0))
            norm_grad = _normalize_image_attr(img_grad.numpy(), sign="all", outlier_perc=2)
            norm_grads.append(norm_grad)
            att_score = results['attention_outputs'][0][i]
            title = 'Attention = ' + str(np.round(att_score,2))
            fig, ax = viz.visualize_image_attr(img_grad.numpy(), original_image, method="blended_heat_map",
                                    alpha_overlay=0.5, sign="absolute_value", show_colorbar=False, title=title)
            out_path = os.path.join('../output/images', (str(i) + '.png'))

        #    fig.savefig(out_path)

        # stack up the activation maps
        grad_output = np.stack(norm_grads)

        assert image.shape == grad_output.shape, 'Output and input size don\'t match'

        grad_output = itk.image_from_array(grad_output)
        raw_name, extension = os.path.splitext(fn)
        new_name = raw_name + '_out' + extension
        out_path = os.path.join('../output/images', new_name)
        itk.imwrite(grad_output, out_path)

        result['metrics'] = wrapped_model.format_readout()
        # TODO verify this is correctly formatted
        result['metrics']['saliency'] = 'filepath:' + out_path
        json_results.append(result)


    # write the results to the output directory
    with open('../output/results.json', 'w') as outfile:
        json.dump(json_results, outfile)