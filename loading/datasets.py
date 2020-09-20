from dataclasses import dataclass
from functools import partial
from typing import List, Any

import pandas
from torch import is_tensor
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import label_binarize
from random import randint
import numpy as np

from loading.img_utils import load_image
from utils.utils import repeat_to_length
import torch


class SingleImageDataset(Dataset):
    # for the use case where we want to classify single images
    def __init__(self, image_frame, root_dir, attribute_specs, return_attribute_dict=False, image_column='ImagePath', transform=None,
                 use_one_channel=False, use_mask=False,strip_folder=False):
        """
        Args:
            image_frame (DataFrame): DataFrame containing image information
            root_dir (string): Directory with all the images.
            attribute (string): The attribute to output. Default: Diagnosis.
        """
        # should return only one, but specified multiple --> error
        # should return multiple, but specified only one --> no error
        if ~return_attribute_dict & (len(attribute_specs) > 1):
            raise ValueError(f'Was asked to return single attribute, but found {len(attribute_specs)}')

        self.image_frame = image_frame
        self.root_dir = root_dir
      #  self.attribute = attribute

        self.transform = transform
        self.use_one_channel = use_one_channel
        self.image_column = image_column
        self.use_mask = use_mask

        self.one_hot_encode_binary = False

      #  self.label_encoder = label_encoder
        self.attribute_specs = attribute_specs
        self.return_attribute_dict = return_attribute_dict
        if strip_folder:
            self.image_frame[image_column] = self.image_frame[image_column].apply(lambda x: os.path.basename(x))

    def __len__(self):
        return len(self.image_frame)

    # def get_all_labels(self):
    #     labels = self.image_frame[self.attribute]
    #     if self.label_encoder:
    #         labels = [self.label_encoder.get_classification_label(l) for l in labels]
    #     return labels

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
        sample = self.image_frame.iloc[idx]

        img_name = os.path.join(self.root_dir, str(sample[self.image_column]))

        load_func = partial(load_image, root_dir=self.root_dir, use_one_channel=self.use_one_channel,
                            use_mask=self.use_mask)
        image = load_func(img_name)

        attribute_dict = {}
        for attribute_spec in self.attribute_specs:
            attribute_label = sample[attribute_spec.name]
            if attribute_spec.target_type != 'regression' and not pandas.isnull(attribute_label):
                attribute_label = attribute_spec.encoder.get_classification_label(attribute_label)
            attribute_dict[attribute_spec.name] = attribute_label

        # attribute_label = sample[self.attribute]
        # if self.label_encoder:
        #     attribute_label = self.label_encoder.get_classification_label(attribute_label)

        if self.transform:
            image = self.transform(image)

        if self.return_attribute_dict:
            return image, attribute_dict  # [attribute_label]
        else:
            return image, list(attribute_dict.values())[0]


class CustomLabelEncoder(object):
    def __init__(self, att_list, one_hot_encode):
        self.classes = att_list
        self.one_hot_encode = one_hot_encode
        self.is_binary = len(self.classes) == 2
        if self.is_binary and self.one_hot_encode:
            # the label binarizer does not properly one-hot encode binary attributes, so we'll cheat
            # add a dummy class and later remove it
            self.classes = self.classes + ['dummy_label']

    def get_classification_label(self, attribute_label):
        if attribute_label not in self.classes:
            raise ValueError(f'Found illegal label {attribute_label}')
        transformed_label = label_binarize([attribute_label], classes=self.classes)
        if 'dummy_label' in self.classes:
            # drop the last column for the dummy class
            transformed_label = transformed_label[:, 0:-1]

        if ~self.one_hot_encode and self.is_binary:
            # return a scalar
            label_to_return = transformed_label[0][0].astype(float)
        elif ~self.one_hot_encode:
            label_to_return = np.argmax(transformed_label[0])
        else:
            # return an array
            label_to_return = transformed_label[0]
        return label_to_return


def default_record_selection(patient):
    return patient.try_closest_fallback_to_latest()

def select_record_for_device(patient, device):
    return patient.select_for_device(device)

class PatientBagDataset(Dataset):
    def __init__(self, patient_list, root_dir,
                 use_pseudopatients, attribute_specs, return_attribute_dict=False, muscles_to_use=None,
                 image_column='ImagePath', transform=None, use_one_channel=True,
                 use_mask=False, stack_images=True, n_images_per_channel=1,
                 merge_left_right=False, record_selection_policy=None, strip_folder=False,
                 enforce_all_images_exist=True):

        # should return only one, but specified multiple --> error
        # should return multiple, but specified only one --> no error
        if ~return_attribute_dict & (len(attribute_specs) > 1):
            raise ValueError(f'Was asked to return single attribute, but found {len(attribute_specs)}')

        self.return_attribute_dict = return_attribute_dict
        self.muscles_to_use = muscles_to_use

        # a policy for record selection
        if not record_selection_policy:
            self.select_record_for_patient = default_record_selection
        else:
            self.select_record_for_patient = record_selection_policy
        self.patients = patient_list
        print(f'Loaded {len(self.patients)} patients.')
        for patient in self.patients:
            self.select_record_for_patient(patient)

        # make pseudopatients for training purposes
        if use_pseudopatients:
            pp_list = []
            for patient in self.patients:
                pps = patient.make_pseudopatients(muscles=self.muscles_to_use, method='each_once')
                pp_list.extend(pps)
            self.patients = pp_list
            print(f'Pseudopatients yielded a total of {len(self.patients)} patients.')
        self.root_dir = root_dir
       # self.attribute = attribute

       #   self.one_hot_encode_binary = False
       #  self.label_encoder = label_encoder

        self.attribute_specs = attribute_specs #make_att_specs()

        self.transform = transform
        self.use_one_channel = use_one_channel
        self.stack_images = stack_images
        self.use_mask = use_mask
        self.image_column = image_column
        # these parameters control what's in the bag
        self.n_images_per_channel = n_images_per_channel
        # whether to bundle up images of the same muscle from left and right
        self.merge_left_right = merge_left_right
        if self.merge_left_right:
            self.grouper = ['Muscle']
        else:
            self.grouper = ['Muscle', 'Side']

        self.strip_folder = strip_folder
        self.enforce_all_images_exist = enforce_all_images_exist

    # def get_all_labels(self):
    #     labels = [patient.attributes[self.attribute] for patient in self.patients]
    #     if self.label_encoder:
    #         labels = [self.label_encoder.get_classification_label(l) for l in labels]
    #     return labels

    def get_total_number_of_images(self):
        total_count = 0
        for patient in self.patients:
            sample = patient.get_selected_record()
            image_frame = sample.image_frame.copy()

            if self.muscles_to_use is not None:
                image_frame = image_frame['Muscle'].isin(self.muscles_to_use)

            # retain n images for each muscle (and side) (n=1 --> the first image)
            image_frame = image_frame.groupby(self.grouper).head(self.n_images_per_channel)
            total_count = total_count + len(image_frame)
        return total_count

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
        patient = self.patients[idx]
        sample = patient.get_selected_record()
        image_frame = sample.image_frame.copy()

        if self.muscles_to_use is not None:
            image_frame = image_frame['Muscle'].isin(self.muscles_to_use)

        # retain n images for each muscle (and side) (n=1 --> the first image)
        image_frame = image_frame.groupby(self.grouper).head(self.n_images_per_channel)

        load_func = partial(load_image, root_dir=self.root_dir, use_one_channel=self.use_one_channel,
                            use_mask=self.use_mask)
        if self.strip_folder:
            image_frame[self.image_column] = image_frame[self.image_column].apply(lambda x: os.path.basename(x))

        if not self.enforce_all_images_exist:
            exists = image_frame[self.image_column].apply(lambda x: os.path.exists(os.path.join(self.root_dir, x)))
            if not exists.all():
                print(f'Missing images {image_frame[~exists][self.image_column]}')
            image_frame = image_frame[exists]
        imgs = image_frame[self.image_column].apply(load_func).to_list()
        if self.transform:
            imgs = [self.transform(image) for image in imgs]
        # stack up the tensors at the end
        if self.stack_images:
            imgs = torch.stack(imgs)

        attribute_dict = {}
        for attribute_spec in self.attribute_specs:
            if attribute_spec.source == 'patient':
                attribute_label = patient.attributes[attribute_spec.name]
            elif attribute_spec.source == 'record':
                attribute_label = sample.meta_info[attribute_spec.name]
            elif attribute_spec.source == 'image':
                attribute_label = image_frame[attribute_spec.name]
            else:
                continue

            if attribute_spec.target_type == 'passthrough':
                pass
            else:
                if attribute_spec.target_type != 'regression' and not pandas.isnull(attribute_label):
                    attribute_label = attribute_spec.encoder.get_classification_label(attribute_label)
            attribute_dict[attribute_spec.name] = attribute_label

        # attribute_label = patient.attributes[self.attribute]
        # # transform into classes if so desired
        # if self.label_encoder:
        #     attribute_label = self.label_encoder.get_classification_label(attribute_label=attribute_label)

        if self.return_attribute_dict:
            return imgs, attribute_dict#[attribute_label]
        else:
            return imgs, list(attribute_dict.values())[0]

    def __len__(self):
        return len(self.patients)


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class Patient(object):
    def __init__(self, records, attributes):
        if len(records) == 0:
            raise ValueError('Patient records cannot be empty.')
        self.records = records
        # the attributes of this patient, e.g. age, diagnosis
        self.attributes = attributes
        # defaults to the first record
        self.record_to_use = 0

    def get_selected_record(self):
        return self.records[self.record_to_use]

    def select_latest(self):
        if len(self.records) > 1:
            dates = []
            for record in self.records:
                date = record.meta_info['RecordingDate']
                dates.append(date)
            self.record_to_use = np.argmax(dates, 0)
            return True
        return False

    def select_closest(self):
        if len(self.records) > 1 and not pandas.isnull(self.attributes['study_date']):
            diffs = []
            for record in self.records:
                date = record.meta_info['RecordingDate']
                diffs.append(np.abs(self.attributes['study_date'] - date))
            self.record_to_use = np.argmin(diffs, 0)
            return True
        return False

    def try_closest_fallback_to_latest(self):
        could_select_closest = self.select_closest()
        if not could_select_closest:
            self.select_latest()

    def select_for_device(self, device):

        if len(self.records) > 1:
            for i, record in enumerate(self.records):
                device_rec = record.meta_info['DeviceInfo']
                if device == device_rec:
                    self.record_to_use = i
                    return True
        return False

    def make_pseudopatients(self, muscles=None, method='each_once', n_limit=100):
        record = self.records[self.record_to_use]
        pseudorecords = make_pseudorecords(record, muscles, method, n_limit)
        pseudopatients = []
        for record in pseudorecords:
            p = Patient([record], self.attributes)
            pseudopatients.append(p)
        return pseudopatients


class PatientRecord(object):
    def __init__(self, r_id, image_frame, meta_info=None):
        self.r_id = r_id
        self.image_frame = image_frame
        # relevant meta information for selecting records
        if meta_info is None:
            meta_info = {}
        self.meta_info = meta_info

    def get_EI_frame(self):
        if 'EIs' not in self.meta_info:
            return pandas.DataFrame()
        muscles = self.meta_info['Muscles_list']
        sides = self.meta_info['Sides']
        ei = self.meta_info['EIs']
        eiz = self.meta_info['EIZ']

        return pandas.DataFrame({'Muscle': muscles, 'Side': sides, 'EI': ei, 'EIZ': eiz})

def make_pseudorecords(record, muscles=None, method='each_once', n_limit=100):
    frame_to_process = record.image_frame.copy()
    if muscles:
        frame_to_process = frame_to_process[frame_to_process['Muscle'].isin(muscles)]
    else:
        muscles = set(frame_to_process['Muscle'])
    # can group here by muscle and side if necessary
    img_dict = frame_to_process.groupby(['Muscle'])['ImagePath'].apply(list).to_dict()
    img_lists = list(img_dict.values())

    # img_lists = [self.images_by_channel[muscle] for muscle in muscles]

    max_len = max([len(img_list) for img_list in img_lists])
    # make sure the lists all have the same length
    img_lists = [img_list if len(img_list) == max_len else repeat_to_length(img_list, max_len) for img_list in
                 img_lists]

    # make all combinations where each image is used exactly once
    if method == 'each_once':
        img_combs = list(zip(*img_lists))
    # sample combinations randomly for efficiency reasons, avoid duplicates
    elif method == 'sample_randomly':
        img_combs = []
        ind_combs = []
        for i in range(n_limit):
            # make a random index for each muscle type
            inds = [randint(0, max_len - 1) for _ in range(len(muscles))]
            # regenerate duplicate combinations if necessary
            while inds in ind_combs:
                inds = [randint(0, max_len - 1) for _ in range(len(muscles))]
            ind_combs.append(inds)
            # select the random sample for each muscle
            selected_samples = [img_lists[list_ind][sample_ind] for list_ind, sample_ind in enumerate(inds)]
            img_combs.append(selected_samples)
    else:
        raise NotImplementedError(f'No such method: {method}')

    pseudorecords = []

    for img_list in img_combs:
        # img_list = [[x] for x in img_list]
        # zip up with muscle_labels
        # images_by_channel = dict(zip(muscles,img_list))
        reduced_image_frame = record.image_frame[record.image_frame['ImagePath'].isin(img_list)].copy()
        p = PatientRecord(record.r_id, reduced_image_frame, record.meta_info)
        pseudorecords.append(p)

    return pseudorecords

class AttributeSpec:

    def __init__(self, name, source, target_type, legal_values=None) -> None:
        super().__init__()
        self.name = name
        self.source = source
        self.target_type = target_type
        self.encoder = None
        if legal_values:
            self.legal_values = legal_values
            self.encoder = CustomLabelEncoder(self.legal_values, one_hot_encode=False)



problem_kind = {'Sex': 'binary', 'Age': 'regression', 'BMI': 'regression',
            'Class': 'binary', 'Muscle': 'passthrough', 'Side': 'passthrough',
                'Age_binned': 'binary', 'BMI_binned': 'binary'} #

problem_source = {'Sex': 'patient', 'Age': 'record', 'BMI': 'record',
            'Class': 'patient', 'Muscle': 'image', 'Side' : 'image',
                  'Age_binned': 'record', 'BMI_binned': 'record'} #

problem_legal_values = {'Class': ['no NMD', 'NMD'], 'Sex': ['F', 'M'],
                        'Age_binned': ['below', 'above'], 'BMI_binned': ['below', 'above']}

def make_att_specs():
    att_spec_dict = {}
    for name, target_type in problem_kind.items():
        legal_values = None
        if name in problem_legal_values:
            legal_values = problem_legal_values[name]
        source = problem_source[name]

        a = AttributeSpec(name,source =source, target_type=target_type, legal_values=legal_values)
        att_spec_dict[name] = a

    return att_spec_dict