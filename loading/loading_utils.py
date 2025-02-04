from loading.loaders import SetSpec

# for each device, we have a test, val and a test set
device_source_mapping = {'ESAOTE_6100': 'umc', 'GE_Logiq_E': 'jhu', 'Philips_iU22': 'umc',
                  'Multiple': 'umc'}

device_splits = {'ESAOTE_6100': ['train', 'val', 'test', ['train', 'val'], 'unlabeled'],
                 'GE_Logiq_E': ['im_muscle_chart'],
                 'Philips_iU22': ['train', 'val', 'test', ['train', 'val'], 'unlabeled'], 'Multiple': ['all']}

# umc_img_root_dict: {'ESAOTE_6100': some_path}
def make_default_img_root_dict(umc_img_root, jhu_img_root):
    img_root_dict = {}
    for device, source in device_source_mapping.items():
        if source == 'umc':
            img_root_dict[device] = umc_img_root
        elif source == 'jhu':
            img_root_dict[device] = jhu_img_root
    return img_root_dict



def make_set_specs(umc_data_path, umc_img_root, jhu_data_path, jhu_img_root):

    label_paths = {'umc': umc_data_path, 'jhu': jhu_data_path}
    img_root_paths = {'umc': umc_img_root, 'jhu': jhu_img_root}
    set_specs = {}
    for device, dataset_type in device_source_mapping.items():
        # get the splits
        splits = device_splits[device]
        label_path = label_paths[dataset_type]
        img_root_path = img_root_paths[dataset_type]
        for split in splits:
            # wrap the single splits in list
            if not isinstance(split, list):
                split = [split]
            spec_name = device + '_' + '+'.join(split)
            set_specs[spec_name] = SetSpec(device, dataset_type, split, label_path, img_root_path)
    return set_specs
