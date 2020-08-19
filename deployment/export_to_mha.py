import os

import itk
import numpy

from baselines import get_default_set_spec_dict
from loading.loaders import get_data_for_spec, make_bag_dataset, make_basic_transform
from loading.datasets import make_att_specs, PatientBagDataset
from loading.datasets import  problem_legal_values
if __name__ == '__main__':
    set_name = 'ESAOTE_6100_val'
    set_spec_dict = get_default_set_spec_dict()
    set_spec = set_spec_dict[set_name]
    patients = get_data_for_spec(set_spec, loader_type='bag', attribute_to_filter='Class',
                                 legal_attribute_values=problem_legal_values['Class'],
                                 muscles_to_use=None)

    att_spec_dict = make_att_specs()


    transform = make_basic_transform(set_spec.device, normalizer_name=None, to_tensor=False, limit_image_size=True)


    ds = PatientBagDataset(patient_list=patients, root_dir=set_spec.img_root_path,
                           attribute_specs=[att_spec_dict['Sex']], transform=transform, use_pseudopatients=False,
                           muscles_to_use=None, use_one_channel=True, return_attribute_dict=False,
                           stack_images=False)

    for i in range(3):
        imgs,y = ds[i]

        imgs = [numpy.asarray(img) for img in imgs]
        x = numpy.stack(imgs)

        im = itk.image_from_array(x)
        os.makedirs('../input/', exist_ok=True)
        file_name = f'../input/patient_esaote_{i}.mha'
        itk.imwrite(im, file_name)