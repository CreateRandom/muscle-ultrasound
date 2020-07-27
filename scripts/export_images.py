import os

from tqdm import tqdm

from baselines import get_default_set_spec_dict, get_mnt_path
from loading.loaders import get_data_for_spec, make_image_exporter

if __name__ == '__main__':

    set_spec_dict = get_default_set_spec_dict()

    specs = [('ESAOTE_6100_train', 'trainA'), ('Philips_iU22_train', 'trainB'), ('ESAOTE_6100_val', 'testA'), ('Philips_iU22_val', 'testB'),
             ('ESAOTE_6100_unlabeled', 'trainA'), ('Philips_iU22_unlabeled', 'trainB'), ('ESAOTE_6100_test', 'ESAOTE_6100_test'),
             ('Philips_iU22_test','Philips_iU22_test')]
    set_spec_name = 'Philips_iU22_val'

    for (set_spec_name, target_name) in specs:
        set_spec = set_spec_dict[set_spec_name]
        images = get_data_for_spec(set_spec, loader_type='image', attribute_to_filter='Image',
                                   muscles_to_use=None)
        mnt_path = get_mnt_path()
        export_path = os.path.join('klaus', 'standard_format', target_name)
        os.makedirs(export_path,exist_ok=True)
        image_exporter = make_image_exporter(images, set_spec.img_root_path, use_one_channel=False, batch_size=8, device=set_spec.device,
                                             export_path=export_path)
        for elem in tqdm(image_exporter):
            pass