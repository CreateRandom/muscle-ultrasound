import os
import socket

from loading.loading_utils import make_set_specs


def get_mnt_path():
    current_host = socket.gethostname()
    if current_host == 'pop-os':
        mnt_path = '/mnt/chansey/'
        if not os.path.ismount(mnt_path):
            return None
    else:
        mnt_path = '/mnt/netcache/diag/'
    return mnt_path


def get_local_set_spec_dict():
    umc_img_root = '/media/klux/Elements/total_patients'
    umc_data_path = '/home/klux/Thesis_2/klaus/data/devices/'
    jhu_data_path = '/home/klux/Thesis_2/klaus/myositis/'
    jhu_img_root = '/home/klux/Thesis_2/klaus/myositis/processed_imgs'
    set_spec_dict = make_set_specs(umc_data_path, umc_img_root, jhu_data_path, jhu_img_root)
    return set_spec_dict


def get_default_set_spec_dict(mnt_path=None, local=False):
    if not mnt_path:
        mnt_path = get_mnt_path()
        print(f'Retrieved mount_path: {mnt_path}')
    # local mode if so desired or nothing could be mounted
    if local or not mnt_path:
        print('Falling back to local path!')
        return get_local_set_spec_dict()
    umc_data_path = os.path.join(mnt_path, 'klaus/data/devices/')
    umc_img_root = os.path.join(mnt_path, 'klaus/total_patients/')
    jhu_data_path = os.path.join(mnt_path, 'klaus/myositis/')
    jhu_img_root = os.path.join(mnt_path, 'klaus/myositis/processed_imgs')
    set_spec_dict = make_set_specs(umc_data_path, umc_img_root, jhu_data_path, jhu_img_root)
    return set_spec_dict