import os

from tqdm import tqdm

from utils.experiment_utils import get_default_set_spec_dict
from loading.loaders import get_data_for_spec, make_image_exporter

if __name__ == '__main__':

    set_spec_dict = get_default_set_spec_dict(local=True)

    specs = [('Philips_iU22_val', 'testB'), ('ESAOTE_6100_train', 'trainA'), ('Philips_iU22_train', 'trainB'), ('ESAOTE_6100_val', 'testA'),
             ('ESAOTE_6100_test', 'ESAOTE_6100_test'), ('Philips_iU22_test','Philips_iU22_test')]
    set_spec_name = 'Philips_iU22_val'

    for (set_spec_name, target_name) in specs:
        set_spec = set_spec_dict[set_spec_name]
        images = get_data_for_spec(set_spec, loader_type='image', attribute_to_filter='Image',
                                   muscles_to_use=None)

        export_path = os.path.join('klaus', 'standard_format', target_name)
        os.makedirs(export_path,exist_ok=True)

        # export the images so the labels can be read in along the side for the semantic consistency loss
        export_images = images.set_index('Image')
        label_export_path = os.path.join('klaus', 'standard_format', target_name, 'labels.pkl')
        export_images.to_pickle(label_export_path)
        dry_run = False
        if not dry_run:
            image_exporter = make_image_exporter(images, set_spec.img_root_path, use_one_channel=False, batch_size=8, device=set_spec.device,
                                                 export_path=export_path)
            for elem in tqdm(image_exporter):
                pass
