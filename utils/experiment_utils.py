import os
import socket

from loading.loading_utils import make_set_specs

# the directory that contains all the data

local_data_path = 'PUT_YOUR_DATA_PATH_HERE'

# the folder that contains the patient data pickle files, separate for each device
umc_data_subpath = 'umc_devices/'
# the folder that contains the patient records with images in them
umc_img_subpath = 'total_patients/'
# the folder that contains the patient file from the JHU dataset
jhu_data_subpath = 'myositis/'
# the folder that contains processed images from the JHU dataset
jhu_img_subpath = 'myositis/processed_imgs'

def get_mnt_path():
    # if a fixed path is specified, use this instead
    if local_data_path:
        return local_data_path

    # dynamically adjust the mnt_path depending on whether we're on the local machine vs on the cluster directly
    current_host = socket.gethostname()
    if current_host == 'pop-os':
        mnt_path = '/mnt/chansey/'
        if not os.path.ismount(mnt_path):
            return None
    else:
        mnt_path = '/mnt/netcache/diag/'
    return mnt_path

def get_default_set_spec_dict(mnt_path=None):
    if not mnt_path:
        mnt_path = get_mnt_path()
        print(f'Retrieved mount_path: {mnt_path}')

    umc_data_path = os.path.join(mnt_path, umc_data_subpath)
    umc_img_root = os.path.join(mnt_path, umc_img_subpath)
    jhu_data_path = os.path.join(mnt_path, jhu_data_subpath)
    jhu_img_root = os.path.join(mnt_path, jhu_img_subpath)
    set_spec_dict = make_set_specs(umc_data_path, umc_img_root, jhu_data_path, jhu_img_root)
    return set_spec_dict