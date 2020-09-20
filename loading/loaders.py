import multiprocessing
import os
from dataclasses import dataclass
from functools import partial
from typing import List, Union

import numpy
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from torchvision.transforms import transforms, CenterCrop, Resize

from loading.datasets import PatientBagDataset, SingleImageDataset, \
    PatientRecord, Patient, AttributeSpec, ConcatDataset, problem_legal_values
from loading.img_utils import FixedHeightCrop, BrightnessBoost, RegressionImageAdjustment


def load_myositis_images(csv_path, muscles_to_use=None) -> pd.DataFrame:
    image_frame = pd.read_csv(csv_path)
    # add image format
    image_frame['ImagePath'] = image_frame['Image2D'].apply(lambda x: str(x) + '.jpg')
    image_frame['Side'] = image_frame['Muscle'].astype(str).apply(lambda x: x[0])
    image_frame['Muscle'] = image_frame['Muscle'].astype(str).apply(lambda x: x[1:])
    image_frame['Muscle'].replace({'D': 'Deltoideus', 'B': 'Biceps brachii','FCR': 'Flexor carpi radialis',
                                   'T': 'Tibialis anterior','R': 'Rectus femoris', 'G': 'Gastrocnemius medial head',
                                    'FDP': 'Flexor digitorum profundus'}, inplace=True)

    image_frame['Class'] = image_frame['Diagnosis'].replace({'N': 'no NMD', 'I': 'NMD', 'P': 'NMD',
                                   'D': 'NMD'})

    if muscles_to_use:
        image_frame = image_frame[image_frame['Muscle'].isin(muscles_to_use)]

    return image_frame

normalizer_params = {'pretrained': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
                     'ESAOTE_6100': {'mean': [0.2396], 'std': [0.1360]},
                     'Philips_iU22': {'mean': [0.1668], 'std': [0.1507]},
                     'GE_Logiq_E': {'mean': [0.302], 'std': [0.828]}}

resize_params = {'ESAOTE_6100': {'center_crop': (480, 503)},
                 'GE_Logiq_E': {'center_crop': (480, 503)},
                 'Philips_iU22': {'fixed_height': (29, 71), 'center_crop': (561, 588), 'resize': (480, 503)}}

resize_params_limited = {'ESAOTE_6100': {'center_crop': (480, 480), 'resize': (224, 224)},
                 'GE_Logiq_E': {'center_crop': (480, 480), 'resize': (224, 224)},
                 'Philips_iU22': {'fixed_height': (29, 71), 'center_crop': (561, 561), 'resize': (224, 224)}}


# Esaote machine standard image size: 480 * 503 (narrower images are 480 * 335)
# Philips: 661 * 588, more variation in the latter domain
# Myo: 479 * 498, narrow images are 498 * 319

# center crop esoate and myo to 480 * 503
# center philips to the corresponding 561 * 588, then scale down
def make_basic_transform(resize_option_name, normalizer_name=None, to_tensor=True, limit_image_size=False,
                         brightness_factor=None, regression_model=None):
    t_list = []
    if resize_option_name:
        if limit_image_size:
            resize_dict = resize_params_limited[resize_option_name]
        else:
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

    # optionally add brightness transform
    if brightness_factor:
        t_list.append(BrightnessBoost(brightness_factor))

    if regression_model:
        t_list.append(RegressionImageAdjustment(regression_model))

    if to_tensor:
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


def umc_to_image_frame(patient_path, record_path, image_path, attribute_to_filter=None,
                       muscles_to_use=None, legal_attribute_values=None, boolean_subset_attribute=None,
                       dropna=False) -> pd.DataFrame:
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
    if boolean_subset_attribute:
        images = images[images[boolean_subset_attribute]]
        print(f'Retained {len(images)} images after filtering on {boolean_subset_attribute}.')

    if attribute_to_filter:
        if legal_attribute_values:
            drop_values = set(images[~images[attribute_to_filter].isin(legal_attribute_values)][attribute_to_filter])
            if drop_values:
                print(f'Replacing {drop_values} with nan.')
                # replace with nan
                images[attribute_to_filter] = images[attribute_to_filter].replace(drop_values,
                                                                                      [numpy.nan] * len(drop_values))

                # patients = images[images[attribute_to_filter].isin(legal_attribute_values)]
                # print(
                #     f'Retained {len(patients)} images after dropping {drop_values}.')
        if dropna:
            images.dropna(subset=[attribute_to_filter], inplace=True)
            print(f'Retained {len(images)} images after dropping null values.')

    return images


def umc_to_patient_list(patient_path, record_path, image_path, attribute_to_filter=None, muscles_to_use=None,
                        legal_attribute_values=None, boolean_subset_attribute=None, dropna=False) -> List[
    Patient]:
    patients = pd.read_pickle(patient_path)

    records = pd.read_pickle(record_path)
    # add binned versions of the attributes
    records['Age_binned'] = pd.cut(records['Age'],bins=[0, 51, 120], labels=['below', 'above'])
    records['BMI_binned'] = pd.cut(records['BMI'], bins=[0, 24.4, 100], labels=['below', 'above'])
    images = pd.read_pickle(image_path)
    patients.set_index('pid', inplace=True)
    print(f'Read {len(patients)} patients, {len(records)} records and {len(images)} images.')
    if boolean_subset_attribute:
        patients = patients[patients[boolean_subset_attribute]]
        records = records[records.pid.isin(set(patients.index))]
        images = images[images['rid'].isin(records.rid)]
        print(
            f'Retained {len(patients)} patients, {len(records)} records and {len(images)} images after retaining based on {boolean_subset_attribute}.')

    if attribute_to_filter:
        if legal_attribute_values:
            drop_values = set(patients[~patients[attribute_to_filter].isin(legal_attribute_values)][attribute_to_filter])
            if drop_values:
                print(f'Replacing {drop_values} with nan.')
                # replace with nan
                patients[attribute_to_filter] = patients[attribute_to_filter].replace(drop_values,
                                                                                      [numpy.nan] * len(drop_values))


        # drop na values if so instructed..
        if dropna:

            patients.dropna(subset=[attribute_to_filter], inplace=True)
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

def collate_bags_to_batch_multi_atts(batch):
    x = [batch[e][0] for e in range(len(batch))]
    att_tensors = {}
    non_tensor_atts = {}
    for e in range(len(batch)):
        att_dict = batch[e][1]
        for k, v in att_dict.items():
            # some attributes might be lists etc (if passed through), store those separately
            try:
                new_v = torch.tensor(v)
                if k not in att_tensors:
                    att_tensors[k] = []
                att_tensors[k].append(new_v)
            except:
                non_tensor_atts[k] = v
    att_tensors = {k: torch.stack(v).unsqueeze(dim=1) for k, v in att_tensors.items()}

    att_tensors = {**att_tensors, **non_tensor_atts}

    n_images_per_bag = torch.tensor([batch[x][0].shape[0] for x in range(len(batch))])
    # concat into n_images * channels * height * width, i.e. squeeze out images per patient
    img_rep = torch.cat(x)
    x = img_rep, n_images_per_bag
    return x, att_tensors


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

def collate_concat_dataset(list_of_batches, base_collate):
    source = []
    target = []
    for batch in list_of_batches:
        source.append(batch[0])
        target.append(batch[1])
    source = base_collate(source)
    target = base_collate(target)
    return source, target

def get_n_cpu():
    n_cpu = multiprocessing.cpu_count()
    # for now cap this at some point to avoid memory issues
    n_cpu = 8 if n_cpu > 8 else n_cpu
    print(f'Using {n_cpu} workers for data loading.')
    return n_cpu


def make_bag_dataset(patients: List[Patient], img_folder, use_one_channel, attribute_specs,
                    transform,return_attribute_dict= False, use_pseudopatients=False, use_mask=False,
                     strip_folder=False, enforce_all_images_exist=True):
    # TODO allow comparison of different methods for using pseudopatients
    ds = PatientBagDataset(patient_list=patients, root_dir=img_folder,
                           attribute_specs= attribute_specs, transform=transform, use_pseudopatients=use_pseudopatients,
                           muscles_to_use=None, use_one_channel=use_one_channel, return_attribute_dict=return_attribute_dict,
                           use_mask=use_mask, strip_folder=strip_folder, enforce_all_images_exist=enforce_all_images_exist)

    print(f'Total number of images in dataset: {ds.get_total_number_of_images()}')

    return ds

def wrap_in_bag_loader(ds, batch_size, pin_memory=False, return_attribute_dict=False, shuffle=True):
    n_cpu = get_n_cpu()
    base_collate_fn = collate_bags_to_batch_multi_atts if return_attribute_dict else collate_bags_to_batch

    if isinstance(ds,ConcatDataset):
        base_collate_fn = partial(collate_concat_dataset, base_collate=base_collate_fn)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=n_cpu, collate_fn=base_collate_fn,
                        pin_memory=pin_memory, drop_last=True)

    return loader

def make_bag_loader(patients: List[Patient], img_folder, use_one_channel, normalizer_name, attribute_specs, batch_size,
                    device, limit_image_size, return_attribute_dict= False,
                    use_pseudopatients=False, pin_memory=False):
    transform = make_basic_transform(device, normalizer_name=normalizer_name, limit_image_size=limit_image_size)
    # TODO allow comparison of different methods for using pseudopatients
    ds = PatientBagDataset(patient_list=patients, root_dir=img_folder,
                           attribute_specs= attribute_specs, transform=transform, use_pseudopatients=use_pseudopatients,
                           muscles_to_use=None, use_one_channel=use_one_channel, return_attribute_dict=return_attribute_dict)

    print(f'Total number of images in bag loader: {ds.get_total_number_of_images()}')
    n_cpu = get_n_cpu()
    collate_fn = collate_bags_to_batch_multi_atts if return_attribute_dict else collate_bags_to_batch
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=n_cpu, collate_fn=collate_fn,
                        pin_memory=pin_memory, drop_last=True)

    return loader

def make_image_exporter(image_frame: pd.DataFrame, img_folder, use_one_channel, batch_size,
                      device, export_path):

    transform = make_basic_transform(device, normalizer_name=None, to_tensor=False, limit_image_size=True)

    def export_batch(batch):
        for image, name in batch:
            image.save(os.path.join(export_path,name))
        return batch
    
    image_spec = AttributeSpec(name='Image',source='image', target_type='regression')

    ds = SingleImageDataset(image_frame=image_frame,root_dir=img_folder,attribute_specs=[image_spec],transform=transform,
                            use_one_channel=use_one_channel)

    print(len(ds))

    n_cpu = get_n_cpu()

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=n_cpu, collate_fn=export_batch,
                        drop_last=True)

    return loader

def make_image_loader(image_frame: pd.DataFrame, img_folder, use_one_channel, normalizer_name, attribute_specs, batch_size,
                      device, limit_image_size, use_mask=False, pin_memory=False, is_multi=False, return_multiple_atts=False):

    transform = make_basic_transform(device, normalizer_name=normalizer_name, limit_image_size=limit_image_size)

    def collate_images_binary(batch):
        batch = default_collate(batch)
        # ensure no single-dim tensors for binary problems, needs to be single dim for multiclass though!
        batch = [elem if len(elem.shape) > 1 else elem.unsqueeze(1) for elem in batch]
        return batch

    ds = SingleImageDataset(image_frame=image_frame, root_dir=img_folder, attribute_specs=attribute_specs,
                            return_attribute_dict= return_multiple_atts, transform=transform,
                            use_one_channel=use_one_channel, use_mask=use_mask)

    n_cpu = get_n_cpu()
    # use default
    collate_fn = None
    if not is_multi:
        collate_fn = collate_images_binary
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=n_cpu, collate_fn=collate_fn, pin_memory=pin_memory,
                        drop_last=True)

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
        # for compatibility, store the patient info in the record as well
        pr = PatientRecord(r_id=pid + '_rec', image_frame=image_frame, meta_info=info.to_dict())
        p = Patient(records=[pr], attributes=info.to_dict())
        return p

    patient_list = info_by_patient.apply(parse_patient, axis=1).to_list()

    return patient_list

@dataclass
class SetSpec:
    device: str
    dataset_type: str
    splits: List[str]
    label_path: str
    img_root_path: str

    def __str__(self):
        return self.device + '/' + '+'.join(self.splits)

def get_classes(data: Union[pd.DataFrame, List[Patient]], attribute):
    if isinstance(data,pd.DataFrame):
        freq_counts = data[attribute].value_counts()
        print(freq_counts)
        return set(data[attribute])
    elif isinstance(data, list):
        atts = [patient.attributes[attribute] for patient in data]
        print(pd.Series(atts).value_counts())
        return set(atts)

def get_data_for_spec(set_spec : SetSpec, loader_type='bag', attribute_to_filter='Class', legal_attribute_values=None, muscles_to_use=None,
                      boolean_subset_attribute=None, dropna_values=True):
    dataset_type = set_spec.dataset_type
    data_path = set_spec.label_path
    device_name = set_spec.device

    if dataset_type == 'umc':
        if loader_type == 'bag':
            all_patients = []
            for split in set_spec.splits:
                # e.g. ESAOTE_6100/train
                set_path = os.path.join(device_name, split)
                to_load = os.path.join(data_path, set_path)

                all_patients.extend(umc_to_patient_list(os.path.join(to_load, 'patients.pkl'),
                                               os.path.join(to_load, 'records.pkl'),
                                               os.path.join(to_load, 'images.pkl'),
                                               attribute_to_filter=attribute_to_filter, boolean_subset_attribute=boolean_subset_attribute,
                                               legal_attribute_values=legal_attribute_values, muscles_to_use=muscles_to_use,
                                               dropna=dropna_values))
            return all_patients

        elif loader_type == 'image':
            all_image_frames = []
            for split in set_spec.splits:
                # e.g. ESAOTE_6100/train
                set_path = os.path.join(device_name, split)
                to_load = os.path.join(data_path, set_path)

                all_image_frames.append(umc_to_image_frame(os.path.join(to_load, 'patients.pkl'),
                                            os.path.join(to_load, 'records.pkl'),
                                            os.path.join(to_load, 'images.pkl'),
                                            attribute_to_filter=attribute_to_filter, boolean_subset_attribute=boolean_subset_attribute,
                                            legal_attribute_values=legal_attribute_values, muscles_to_use=muscles_to_use,
                                            dropna=dropna_values))

            return pd.concat(all_image_frames)

        raise ValueError(f'Unknown loader type {loader_type}')

    elif dataset_type == 'jhu':
        all_elems = []
        for split in set_spec.splits:
            csv_name = split + '.csv'
            to_load = os.path.join(data_path, csv_name)
            image_frame = load_myositis_images(to_load, muscles_to_use=muscles_to_use)

            if legal_attribute_values:
                drop_values = set(image_frame[~image_frame[attribute_to_filter].isin(legal_attribute_values)][attribute_to_filter])
                if drop_values:
                    image_frame = image_frame[image_frame[attribute_to_filter].isin(legal_attribute_values)]
                    print(
                        f'Retained {len(image_frame)} images after dropping {drop_values}.')

            if loader_type == 'bag':
                patient_list = image_frame_to_patients(image_frame)
                all_elems.extend(patient_list)

            elif loader_type == 'image':
                all_elems.append(image_frame)

            else:
                raise ValueError(f'Unknown loader type {loader_type}')

        if loader_type == 'bag':
            return all_elems
        else:
            return pd.concat(all_elems)

    else:
        raise ValueError(f'Data type {dataset_type} not understood.')
