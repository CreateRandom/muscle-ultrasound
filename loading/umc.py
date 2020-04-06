from functools import partial

import PIL
from torchvision.datasets import DatasetFolder
import torchvision.transforms as transforms

import pydicom

from scipy.io import loadmat
import numpy as np


def make_bbox_from_lines(lines):
    """
    Takes in a 2D array that represents a list of 2D points (one line per dimension)
    and returns a bounding box that wraps around the points.
    :param lines: A 2D array that represents a ROI
    :return: a bbox around the ROI
    """
    y_min = int(np.floor(min(lines[0])))
    y_max = int(np.ceil(max(lines[0])))
    x_min = int(np.floor(min(lines[1])))
    x_max = int(np.ceil(max(lines[1])))
    return x_min, x_max, y_min, y_max


def create_mask(mat_file_path, im_arr):
    """
    Creates a mask that represents the ROI in the ultrasound images from mat file
    that contains drawn lines.
    :param mat_file_path: The path to the mat file
    :param im_arr: The image array
    :return: A mask that represents the ROI, True if the pixel is inside the ROI
    """
    mat_file = loadmat(mat_file_path)
    lines = mat_file['roi']
    # todo think about the bounding box
    x_min, x_max, y_min, y_max = make_bbox_from_lines(lines)
    mask = np.zeros_like(im_arr)
    mask[x_min:x_max, y_min:y_max] = 1
    mask = mask.astype(bool)

    return mask


def load_and_crop_dicom(dicom_path, load_mask=False, use_one_channel=False):
    """
    A loader function for reading ultrasound dicoms and optionally masks stored on the same path
    :param dicom_path: A dicom_path
    :param load_mask: Whether we should try loading the mask
    :return: An image array with one channel
    """
    f = pydicom.dcmread(dicom_path)
    im_arr = f.pixel_array
    # we only care for one channel, as the others are redundant
    if use_one_channel:
        im_arr = im_arr[:, :, 1]
    # load the mask from the mat file of the same name
    if load_mask:
        mat_path = dicom_path.strip('.dcm') + '.mat'
        try:
            mask = create_mask(mat_path, im_arr)
            im_arr[~mask] = 0
        except:
            print('WARNING: Could not retrieve mask mat file, not applying mask.')

    # TODO better cropping
    im = im_arr[78:552, 241:743]
    im = PIL.Image.fromarray(im)
    return im


def make_umc_set(base_path, load_mask=False, use_one_channel=False, normalize=True):
    """

    :param base_path: A path that contains a folder for each class, which in turn contains dicom files
    and optionally mat files with ROIs to be used as mask
    :param load_mask: Whether mat files should be loaded and used as masks
    :return: A torch.utils.data.Dataset
    """
    # for now, use a standard class with a dicom loader, might want to come up with a custom format
    loader = partial(load_and_crop_dicom,load_mask=load_mask, use_one_channel=use_one_channel)

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