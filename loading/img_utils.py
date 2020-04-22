import PIL
import PIL.Image

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
            mask = create_mask(mat_path, im_arr)
            im_arr[~mask] = 0
        except:
            print('WARNING: Could not retrieve mask mat file, not applying mask.')

    im = PIL.Image.fromarray(im_arr)
    return im


def load_img(img_name, use_one_channel=False):
    # one or three channels
    mode = 'L' if use_one_channel else 'RGB'
    # load as PIL image
    with open(img_name, 'rb') as f:
        image = PIL.Image.open(f)
        image = image.convert(mode)
    return image


class AugmentWrapper(object):
    def __init__(self, augment):
        self.augment = augment

    def __call__(self, img):
        return self.augment.augment_image(np.array(img))