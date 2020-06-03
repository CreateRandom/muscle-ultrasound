import multiprocessing
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import transforms, CenterCrop, Resize

from loading.datasets import PatientBagDataset, SingleImageDataset, \
    parse_image_level_frame, PatientRecord, Patient
from loading.img_utils import FixedHeightCrop
from utils.utils import compute_normalization_parameters

def load_myositic_patients(csv_path):
    meta_frame = pd.read_csv(csv_path, index_col=0)
    # add image format
    meta_frame['Image2D'] = meta_frame['Image2D'].apply(lambda x: str(x) + '.jpg')
    meta_frame['Side'] = meta_frame['Muscle'].astype(str).apply(lambda x: x[0])
    meta_frame['Muscle'] = meta_frame['Muscle'].astype(str).apply(lambda x: x[1:])

    return parse_image_level_frame(meta_frame)

def make_myositis_loaders(train_path, val_path, img_folder, use_one_channel, normalizer_name, attribute, batch_size,
                          use_pseudopatients=False, is_classification=True, pin_memory=False):
    train_transform = make_basic_transform_new('GE_Logiq_E', normalizer_name=normalizer_name)

    train_patients = load_myositic_patients(train_path)

    train_loader = make_patient_bag_loader(train_patients, img_folder,
                                           attribute=attribute, transform=train_transform,
                                           batch_size=batch_size,
                                           use_pseudopatients=use_pseudopatients,
                                           use_one_channel=use_one_channel,
                                           is_classification=is_classification,
                                           pin_memory=pin_memory)

    val_transform = make_basic_transform_new('GE_Logiq_E', normalizer_name=normalizer_name)

    val_patients = load_myositic_patients(val_path)

    val_loader = make_patient_bag_loader(val_patients, img_folder,
                                         attribute=attribute, transform=val_transform,
                                         batch_size=batch_size,
                                         use_pseudopatients=False,
                                         use_one_channel=use_one_channel,
                                         is_classification=is_classification,
                                         pin_memory=pin_memory)

    return train_loader, val_loader

normalizer_params = {'pretrained': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
                     'ESAOTE_6100': {'mean': [0.241], 'std': [0.141]},
                     'GE_Logiq_E': {'mean': [0.302], 'std': [0.828]}}

resize_params = {'ESAOTE_6100': {'center_crop': (480,503)},
                 'GE_Logiq_E': {'center_crop': (480,503)},
                 'Philips_iU22': {'fixed_height': (29, 71), 'center_crop': (561,588),'resize': (480,503)}}

# Esaote machine standard image size: 480 * 503 (narrower images are 480 * 335)
# Philips: 661 * 588, more variation in the latter domain
# Myo: 479 * 498, narrow images are 498 * 319

# center crop esoate and myo to 480 * 503
# center philips to the corresponding 561 * 588, then scale down
def make_basic_transform_new(resize_option_name, normalizer_name=None):
    t_list = []

    resize_dict = resize_params[resize_option_name]

    if 'fixed_height' in resize_dict:
        resize_tuple = resize_dict['fixed_height']
        t_list.append(FixedHeightCrop(remove_top=resize_tuple[0], remove_bottom=resize_tuple[1]))
    if 'center_crop' in resize_dict:
        img_size = resize_dict['center_crop']
        center_crop = CenterCrop(img_size)
        t_list.append(center_crop)
    if 'resize' in resize_dict:
        img_size = resize_dict['resize']
        resize = Resize(img_size)
        t_list.append(resize)

    # toTensor automatically scales between 0 and 1
    t_list.append(transforms.ToTensor())

    # pixel values in 0-1 range to z-scores
    if normalizer_name:
        if normalizer_name not in normalizer_params:
            raise ValueError(f'Unknown normalizer name: {normalizer_name}')
        params = normalizer_params[normalizer_name]
        normalize_func = transforms.Normalize(**params)

        t_list.append(normalize_func)

    return transforms.Compose(t_list)


def preprocess_umc_patients(csv_path, muscles_to_use):
    meta_frame = pd.read_csv(csv_path)
    # drop images with no ROI annotation
    meta_frame = meta_frame.dropna(subset=['min_h_roi'])
    print(f'Found a total of {len(meta_frame)} images.')

    # use all muscle channels by default
    if not muscles_to_use:
        muscles_to_use = set(meta_frame['Muscle'])
    meta_frame = meta_frame[meta_frame['Muscle'].isin(muscles_to_use)]

    print(f'Retained a total of {len(meta_frame)} images.')

    # merge folder and file path
    meta_frame['ImagePath'] = meta_frame.apply(lambda x: os.path.join(str(x['folder_name']), str(x['Image'])), axis=1)
    meta_frame.drop(inplace=True, columns=['folder_name', 'Image'])

    return meta_frame

def preprocess_umc_patients_new(patient_path, record_path, image_path, attribute_to_use=None, muscles_to_use=None):
    patients = pd.read_pickle(patient_path)
    patients.set_index('pid', inplace=True)
    records = pd.read_pickle(record_path)
    images = pd.read_pickle(image_path)

    print(f'Read {len(patients)} patients, {len(records)} records and {len(images)} images.')
    if attribute_to_use:
        patients.dropna(subset=[attribute_to_use], inplace=True)
        records = records[records.pid.isin(set(patients.index))]
        images = images[images['rid'].isin(records.rid)]
        print(f'Retained {len(patients)} patients, {len(records)} records and {len(images)} images after dropping null values.')

    if muscles_to_use:
        images = images[images['Muscle'].isin(muscles_to_use)]
        # drop records that don't have this muscle
        # drop patients that have an associated record
        records = records[records.rid.isin(set(images['rid']))]
        patients = patients[patients.index.isin(set(records['pid']))]
        print(f'Retained {len(records)} records from {len(patients)} patients after muscle selection for {muscles_to_use}')
    images['ImagePath'] = images.apply(lambda x: os.path.join(str(x['folder_name']), str(x['Image'])), axis=1)
    images = images[['rid','Muscle','Side','ImagePath']]

    grouped_images = images.groupby('rid')
    images_by_rid = {rid: frame for rid, frame in grouped_images}

    converted_patients = []

    def parse_record(record):
        if record['rid'] not in images_by_rid:
            return None

        patient_images = images_by_rid[record['rid']]

        pr = PatientRecord(r_id=record['rid'], meta_info=record.to_dict(), image_frame=patient_images)
        return pr

    records['converted_record'] = records.apply(parse_record, axis=1).to_list()

    groups = records.groupby('pid')
    for pid, patient_records in groups:
        patient = patients.loc[pid]
        # patient_records['converted_records'] = patient_records.apply(parse_record,axis=1).to_list()
        record_list = patient_records['converted_record'].to_list()
        p = Patient(attributes=patient.to_dict(), records=record_list)
        converted_patients.append(p)

    return converted_patients

# TODO move to some other place
def collate_bags_to_batch(batch):
    x = [batch[e][0] for e in range(len(batch))]
    y = [torch.tensor(batch[e][1]) for e in range(len(batch))]
    n_images_per_bag = torch.tensor([batch[x][0].shape[0] for x in range(len(batch))])
    # concat into n_images * channels * height * width, i.e. squeeze out images per patient
    img_rep = torch.cat(x)
    # this is the inverse op we need to perform on the embeddings
    # back = torch.split(img_rep, n_images_per_bag)
    # thus, pass along how many images are in each bag
    x = img_rep, n_images_per_bag
    y = torch.stack(y)
    return x, y

def make_patient_bag_loader(patients, root_folder, attribute, transform, batch_size, use_pseudopatients,
                            use_one_channel, is_classification, pin_memory):

    ds = PatientBagDataset(patient_list=patients, root_dir=root_folder,
                           attribute=attribute, transform=transform, use_pseudopatients=use_pseudopatients,
                           muscles_to_use=None, use_one_channel=use_one_channel, is_classification=is_classification)

    n_cpu = multiprocessing.cpu_count()
    print(f'Using {n_cpu} workers for data loading.')
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=n_cpu, collate_fn=collate_bags_to_batch,
                        pin_memory=pin_memory)
    return loader


def select_muscles_to_use(csv_path, top_n=4):
    meta_frame = pd.read_csv(csv_path)
    # drop images with no ROI annotation
    meta_frame = meta_frame.dropna(subset=['min_h_roi'])
    x = meta_frame['Muscle'].value_counts().nlargest(top_n)
    muscles = set(x.index)
    return muscles


def make_umc_loader(patients, img_folder, use_one_channel, normalizer_name, attribute, batch_size, device,
                    use_pseudopatients=False, is_classification=True, pin_memory=False):

    train_transform = make_basic_transform_new(device, normalizer_name=normalizer_name)

    # TODO allow comparison of different methods for using pseudopatients
    loader = make_patient_bag_loader(patients, img_folder,
                                     attribute=attribute, transform=train_transform, batch_size=batch_size,
                                     use_pseudopatients=use_pseudopatients, use_one_channel=use_one_channel,
                                     is_classification=is_classification,
                                     pin_memory=pin_memory)

    return loader


def compute_empirical_mean_and_std(csv_path, root_folder, transform, subsample=None, seed=None):
    meta_frame = preprocess_umc_patients(csv_path, muscles_to_use=None)
    if subsample:
        meta_frame = meta_frame.sample(n=subsample, random_state=seed)
    ds = SingleImageDataset(meta_frame, root_folder, attribute='Class', transform=transform, use_one_channel=True,
                            image_column='ImagePath')
    n_channels = 1
    mean, std = compute_normalization_parameters(ds, n_channels)
    return mean, std
