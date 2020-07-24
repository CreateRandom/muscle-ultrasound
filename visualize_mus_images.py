import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from loading.datasets import PatientBagDataset, make_att_specs
from loading.loaders import make_basic_transform, umc_to_patient_list


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

device = 'ESAOTE_6100'
mnt_path = '/mnt/chansey/'
umc_data_path = os.path.join(mnt_path, 'klaus/data/devices/')


base_path = os.path.join(umc_data_path,device, 'train')
patients = umc_to_patient_list(os.path.join(base_path, 'patients.pkl'),
                                    os.path.join(base_path, 'records.pkl'),
                                    os.path.join(base_path, 'images.pkl'), dropna=False)
att_spec_dict = make_att_specs()

transform = make_basic_transform(device, limit_image_size=True)
ds = PatientBagDataset(patient_list=patients, root_dir='/mnt/chansey/klaus/total_patients/', use_pseudopatients=False,
                       attribute_specs=[att_spec_dict['Sex']], transform=transform,stack_images=False, use_mask=False)
for i in range(5):
    x, y = ds[i]
    for elem in x:
        print(elem.shape)
      #  plt.imshow(elem.squeeze(), cmap='gray')
       # plt.show()
    fig = plot_patient_images(x)
    filename = device + '_' + str(i) + '_pretrained'
    filepath = os.path.join('example_images', filename)
    plt.savefig(filepath)


