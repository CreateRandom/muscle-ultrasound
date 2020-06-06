import multiprocessing
import os
from dataclasses import dataclass
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from torchvision.transforms import transforms, CenterCrop, Resize

from loading.datasets import PatientBagDataset, SingleImageDataset, \
    PatientRecord, Patient
from loading.img_utils import FixedHeightCrop
from utils.utils import compute_normalization_parameters


def load_myositis_images(csv_path):
    image_frame = pd.read_csv(csv_path)
    # add image format
    image_frame['ImagePath'] = image_frame['Image2D'].apply(lambda x: str(x) + '.jpg')
    image_frame['Side'] = image_frame['Muscle'].astype(str).apply(lambda x: x[0])
    image_frame['Muscle'] = image_frame['Muscle'].astype(str).apply(lambda x: x[1:])

    return image_frame

normalizer_params = {'pretrained': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
                     'ESAOTE_6100': {'mean': [0.241], 'std': [0.141]},
                     'GE_Logiq_E': {'mean': [0.302], 'std': [0.828]}}

resize_params = {'ESAOTE_6100': {'center_crop': (480, 503)},
                 'GE_Logiq_E': {'center_crop': (480, 503)},
                 'Philips_iU22': {'fixed_height': (29, 71), 'center_crop': (561, 588), 'resize': (480, 503)}}


# Esaote machine standard image size: 480 * 503 (narrower images are 480 * 335)
# Philips: 661 * 588, more variation in the latter domain
# Myo: 479 * 498, narrow images are 498 * 319

# center crop esoate and myo to 480 * 503
# center philips to the corresponding 561 * 588, then scale down
def make_basic_transform(resize_option_name, normalizer_name=None):
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


def umc_to_image_frame(patient_path, record_path, image_path, attribute_to_use=None,
                       muscles_to_use=None) -> pd.DataFrame:
    patients = pd.read_pickle(patient_path)

    records = pd.read_pickle(record_path)
    images = pd.read_pickle(image_path)

    print(f'Found a total of {len(images)} images.')

    if muscles_to_use:
        images = images[images['Muscle'].isin(muscles_to_use)]
        print(f'Retained {len(images)} images after muscle selection for {muscles_to_use}')

    # the pids in the image frame are broken due to Matlab's table formatting
    images.drop(columns=['pid'], inplace=True)
    record_corrector = records[['rid', 'pid']]
    images = pd.merge(images, record_corrector, how='left')

    images['ImagePath'] = images.apply(lambda x: os.path.join(str(x['folder_name']), str(x['Image'])), axis=1)

    # merge in patient level information
    images = pd.merge(images, patients, on=['pid'], how='left')

    if attribute_to_use:
        patients.dropna(subset=[attribute_to_use], inplace=True)
        images = images[images['pid'].isin(patients.pid)]
        print(f'Retained {len(images)} images from {len(patients)} patients after dropping null values.')

    return images


def umc_to_patient_list(patient_path, record_path, image_path, attribute_to_use=None, muscles_to_use=None) -> List[
    Patient]:
    patients = pd.read_pickle(patient_path)

    records = pd.read_pickle(record_path)
    images = pd.read_pickle(image_path)
    patients.set_index('pid', inplace=True)
    print(f'Read {len(patients)} patients, {len(records)} records and {len(images)} images.')
    if attribute_to_use:
        patients.dropna(subset=[attribute_to_use], inplace=True)
        records = records[records.pid.isin(set(patients.index))]
        images = images[images['rid'].isin(records.rid)]
        print(
            f'Retained {len(patients)} patients, {len(records)} records and {len(images)} images after dropping null values.')

    if muscles_to_use:
        images = images[images['Muscle'].isin(muscles_to_use)]
        # drop records that don't have this muscle
        # drop patients that have an associated record
        records = records[records.rid.isin(set(images['rid']))]
        patients = patients[patients.index.isin(set(records['pid']))]
        print(
            f'Retained {len(records)} records from {len(patients)} patients after muscle selection for {muscles_to_use}')
    images['ImagePath'] = images.apply(lambda x: os.path.join(str(x['folder_name']), str(x['Image'])), axis=1)
    images = images[['rid', 'Muscle', 'Side', 'ImagePath']]

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

def get_n_cpu():
    n_cpu = multiprocessing.cpu_count()
    # for now cap this at some point to avoid memory issues
    n_cpu = 8 if n_cpu > 8 else n_cpu
    print(f'Using {n_cpu} workers for data loading.')
    return n_cpu

def make_bag_loader(patients: List[Patient], img_folder, use_one_channel, normalizer_name, attribute, batch_size,
                    device,
                    use_pseudopatients=False, is_classification=True, pin_memory=False):
    transform = make_basic_transform(device, normalizer_name=normalizer_name)
    # TODO allow comparison of different methods for using pseudopatients
    ds = PatientBagDataset(patient_list=patients, root_dir=img_folder,
                           attribute=attribute, transform=transform, use_pseudopatients=use_pseudopatients,
                           muscles_to_use=None, use_one_channel=use_one_channel, is_classification=is_classification)
    n_cpu = get_n_cpu()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=n_cpu, collate_fn=collate_bags_to_batch,
                        pin_memory=pin_memory)

    return loader

def make_image_loader(image_frame: pd.DataFrame, img_folder, use_one_channel, normalizer_name, attribute, batch_size,
                    device, is_classification=True, pin_memory=False):

    transform = make_basic_transform(device, normalizer_name=normalizer_name)

    def collate_images(batch):
        batch = default_collate(batch)
        # ensure no single-dim tensors (always batch_size first)
        batch = [elem if len(elem.shape) > 1 else elem.unsqueeze(1) for elem in batch]
        return batch

    ds = SingleImageDataset(image_frame=image_frame,root_dir=img_folder,attribute=attribute,transform=transform,
                            use_one_channel=use_one_channel,is_classification=is_classification)

    n_cpu = get_n_cpu()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=n_cpu, collate_fn=collate_images, pin_memory=pin_memory)

    return loader

def image_frame_to_patients(frame) -> List[Patient]:
    image_columns = ['ImagePath', 'Muscle', 'Side']
    fixed_info_columns = frame.columns[~frame.columns.isin(image_columns)]
    patient_info_frame = frame[fixed_info_columns]
    info_by_patient = patient_info_frame.groupby('PatientID').first()

    image_frame = frame[['PatientID', 'ImagePath', 'Muscle', 'Side']]
    images_by_patient = image_frame.groupby('PatientID')
    images_by_patient = {pid: frame for pid, frame in images_by_patient}

    def parse_patient(patient):
        pid = patient.name
        info = info_by_patient.loc[pid].copy()
        # create a record from the image table
        image_frame = images_by_patient[pid]
        pr = PatientRecord(r_id=pid + '_rec', image_frame=image_frame)
        p = Patient(records=[pr], attributes=info.to_dict())
        return p

    patient_list = info_by_patient.apply(parse_patient, axis=1).to_list()

    return patient_list

@dataclass
class SetSpec:
    device: str
    dataset_type: str
    split: str
    label_path: str
    img_root_path: str

    def __str__(self):
        return self.device + '/' + self.split

def get_data_for_spec(set_spec : SetSpec, loader_type='bag', attribute='Class'):
    dataset_type = set_spec.dataset_type
    data_path = set_spec.label_path
    device_name = set_spec.device

    if dataset_type == 'umc':
        # e.g. ESAOTE_6100/train
        set_path = os.path.join(device_name, set_spec.split)
        to_load = os.path.join(data_path, set_path)

        if loader_type == 'bag':
            patients = umc_to_patient_list(os.path.join(to_load, 'patients.pkl'),
                                           os.path.join(to_load, 'records.pkl'),
                                           os.path.join(to_load, 'images.pkl'),
                                           attribute_to_use=attribute)
            return patients

        elif loader_type == 'image':

            images = umc_to_image_frame(os.path.join(to_load, 'patients.pkl'),
                                        os.path.join(to_load, 'records.pkl'),
                                        os.path.join(to_load, 'images.pkl'),
                                        attribute_to_use=attribute)

            return images

        raise ValueError(f'Unknown loader type {loader_type}')

    elif dataset_type == 'jhu':
        csv_name = set_spec.split + '.csv'
        to_load = os.path.join(data_path, csv_name)
        image_frame = load_myositis_images(to_load)

        if loader_type == 'bag':
            return image_frame_to_patients(image_frame)

        elif loader_type == 'image':
            return image_frame

        raise ValueError(f'Unknown loader type {loader_type}')

    else:
        raise ValueError(f'Data type {dataset_type} not understood.')
