import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

from baselines.domain_mapping import get_brightness_factor, get_lr_model, get_mapped_path
from loading.datasets import PatientBagDataset, make_att_specs, problem_legal_values, select_record_for_device
from loading.loaders import make_basic_transform, umc_to_patient_list, get_data_for_spec
from utils.experiment_utils import get_default_set_spec_dict, get_mnt_path


def plot_patient_images(x):
    # patient in format n_image * 1 * x * y
    n_image = len(x)
    grid_size = int(np.ceil(np.sqrt(n_image)))
    fig = plt.figure(figsize=(grid_size * 2, grid_size * 2),dpi=300)

    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(grid_size, grid_size),  # creates 2x2 grid of axes
                     label_mode='l',
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    for i in range(n_image):
        ax = grid[i]
        im = x[i].squeeze()
        ax.imshow(im, cmap='gray')#, vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])

   # plt.axis('off')
   # plt.grid(b=None)
    return fig

# utility script for exporting and inspecting muscle images

if __name__ == '__main__':
    
    source_device = 'ESAOTE_6100'
    target_device = 'Philips_iU22'
    export_path = '../example_images'
    
    matplotlib.rcParams['savefig.pad_inches'] = 0
    
    set_spec_dict = get_default_set_spec_dict()
    device_to_use = target_device + '_val'
    set_spec = set_spec_dict[device_to_use]
    
    patients = get_data_for_spec(set_spec, loader_type='bag', attribute_to_filter='Class',
                                 legal_attribute_values=problem_legal_values['Class'],
                                 muscles_to_use=None)
    
    att_spec_dict = make_att_specs()
    adjustment = ''# mapped_images', 'brightness' etc.
    
    
    transform_dict = {'resize_option_name': target_device, 'limit_image_size': False}
    if adjustment == 'brightness':
        transform_dict['brightness_factor'] = get_brightness_factor(source_device, target_device)
    elif adjustment == 'regression':
        transform_dict['regression_model'] = get_lr_model(source_device, target_device)
    elif adjustment == 'mapped_images':
        transform_dict['resize_option_name'] = None
    
    transform = make_basic_transform(**transform_dict)
    mask = False
    
    root_dir = set_spec.img_root_path
    strip_folder = False
    if adjustment == 'mapped_images':
        method = 'standard_cyclegan'
        root_dir = get_mapped_path(source_device, target_device, method)
        strip_folder = True
        adjustment = adjustment + '_' + method
    
    ds = PatientBagDataset(patient_list=patients, root_dir=root_dir, use_pseudopatients=False,
                           attribute_specs=[att_spec_dict['Class']], transform=transform,stack_images=False, use_mask=mask,
                           n_images_per_channel=3,record_selection_policy=None, strip_folder=strip_folder)
    
    # export settings
    export_single_images = True
    # if True, export histogram of image instead of image itself
    export_hist = False
    adjustment_string = adjustment if adjustment else ''
    # export the first five patients
    for i in [0,1,2,3,4]:
        x, y = ds[i]
        # export the first 10 images per patient
        x = x[0:10]
        if export_single_images:
            for j, elem in enumerate(x):
                plt.figure(figsize=(6,6))
                print(elem.shape)
                plt.imshow(elem.squeeze(), cmap='gray')
                plt.yticks(([]))
                plt.xticks(([]))
                if adjustment_string:
                    suffix = '_' + adjustment_string
                else:
                    suffix = ''
                filename = target_device  + '_' + str(i) + '_' + str(j) + suffix
                if mask:
                    filename = filename + '_mask'
                filepath = os.path.join(export_path, filename)
                plt.savefig(filepath,dpi=200)
                plt.close()
                if export_hist:
                    filename = filename + '_hist'
                    im = elem.squeeze().flatten()
                    im = im[im > 0]
                    im = im * 255
                    plt.hist(im,range=(0,255),bins=255)
                    plt.axvline(im.mean(), color='k', linestyle='dashed', linewidth=1)
                    filepath = os.path.join(export_path, filename)
                    plt.savefig(filepath, dpi=200)
                    plt.close()
        # make a combined plot with all muscles for a given patient
        fig = plot_patient_images(x)
        filename = target_device + '_' + str(i) + '_pretrained'
        filepath = os.path.join(export_path, filename)
        plt.savefig(filepath)