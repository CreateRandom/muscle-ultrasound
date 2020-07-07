import os
from functools import partial

import PIL
import PIL.Image

import pydicom
from PIL import Image

from scipy.io import loadmat
import numpy as np
from skimage.draw import polygon2mask
import torchvision.transforms.functional as TF

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


def create_mask(mat_file_path, im_shape):
    """
    Creates a mask that represents the ROI in the ultrasound images from mat file
    that contains drawn lines.
    :param mat_file_path: The path to the mat file
    :param im_shape: The image size
    :return: A mask that represents the ROI, True if the pixel is inside the ROI
    """
    mat_file = loadmat(mat_file_path)
    if 'r' in mat_file:
        lines = mat_file['r']['roi'][0][0]
    elif 'roi' in mat_file:
        lines = mat_file['roi']
    else:
        return None
    return polygon2mask(im_shape,lines.transpose())


def load_dicom(dicom_path, load_mask=False, use_one_channel=False):
    """
    A loader function for reading ultrasound dicoms and optionally masks stored on the same path
    :param dicom_path: A dicom_path
    :param load_mask: Whether we should try loading the mask
    :return: An image array with one channel
    """
    f = pydicom.dcmread(dicom_path)
    im_arr = f.pixel_array
    # stack up the one channel to normal "RGB"
    if not use_one_channel:
        im_arr = np.stack((im_arr,im_arr,im_arr),axis=2)
    # load the mask from the mat file of the same name
    if load_mask:
        mat_path = dicom_path + '.mat'
        try:
            mask = create_mask(mat_path, im_arr.shape)
            im_arr[~mask] = 0
        except:
            print('WARNING: Could not retrieve mask mat file, not applying mask.')

    im = PIL.Image.fromarray(im_arr)
    return im


def load_pil_img(img_name, use_one_channel=False):
    # one or three channels
    mode = 'L' if use_one_channel else 'RGB'
    # load as PIL image
    with open(img_name, 'rb') as f:
        image = PIL.Image.open(f)
        image = image.convert(mode)
    return image


class FixedHeightCrop(object):

    def __init__(self, remove_top, remove_bottom):
        self.remove_top = remove_top
        self.remove_bottom = remove_bottom

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """

        return img.crop((0, self.remove_top + 1, img.width, img.height - self.remove_bottom + 1))

    def __repr__(self):
        return self.__class__.__name__ + '(remove_top={0}, remove_bottom={1})'.format(self.remove_top, self.remove_bottom)

class BrightnessBoost:
    """Rotate by one of the given angles."""

    def __init__(self, brightness_factor):
        self.brightness_factor = brightness_factor

    def __call__(self, x):
        return TF.adjust_brightness(x, self.brightness_factor)



loader_funcs = {'.png': load_pil_img, '.jpg': load_pil_img, '.dcm': load_dicom}


def load_image(img, root_dir, use_one_channel, use_mask):
    name = str(img)
    img_name = os.path.join(root_dir, name)
    raw_name, extension = os.path.splitext(img_name)
    loader_func = partial(loader_funcs[extension], use_one_channel=use_one_channel)
    image = loader_func(img_name)
    if use_mask:
        # also optionally try loading the mask
        try:
            mat_file_path = raw_name + '.dcm.mat'
            mask = create_mask(mat_file_path, image.size)
            image2 = np.array(image)
            mask = mask.transpose()
            image2[~mask] = 0
            image = Image.fromarray(image2)
        except:
            print(f'Error loading mask for {name}')
    return image