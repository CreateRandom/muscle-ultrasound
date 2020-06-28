import os

from tqdm import tqdm

from baselines import get_default_set_spec_dict, get_mnt_path
from loading.loaders import get_data_for_spec, make_image_exporter

if __name__ == '__main__':

    set_spec_dict = get_default_set_spec_dict()
    set_spec_name = 'ESAOTE_6100_train'
    set_spec = set_spec_dict[set_spec_name]
    images = get_data_for_spec(set_spec, loader_type='image', attribute='Image',
                                 muscles_to_use=None)
    images = images[0:16]
    mnt_path = get_mnt_path()
    export_path = os.path.join(mnt_path, 'klaus', 'standard_format', set_spec_name)
    os.makedirs(export_path,exist_ok=True)
    image_exporter = make_image_exporter(images, set_spec.img_root_path, use_one_channel=False, batch_size=8, device=set_spec.device,
                                         export_path=export_path)
    for elem in tqdm(image_exporter):
        pass