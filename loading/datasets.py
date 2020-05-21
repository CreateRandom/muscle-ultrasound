from collections import Counter
from copy import deepcopy
from functools import partial

from torch import is_tensor
import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import PIL.Image
from random import randint

from loading.img_utils import load_dicom, load_img
from utils.utils import repeat_to_length
import torch

class SingleImageDataset(Dataset):
    # for the use case where we want to classify single images as to the patient's diagnosis
    def __init__(self, meta_frame, root_dir, attribute='Diagnosis', transform=None,
                 use_one_channel=False, image_column='Image2D'):
        """
        Args:
            meta_frame (DataFrame): DataFrame containing image information
            root_dir (string): Directory with all the images.
            attribute (string): The attribute to output. Default: Diagnosis.
        """
        self.meta_frame = meta_frame
        self.root_dir = root_dir
        self.attribute = attribute
        self.encoder = LabelEncoder()
        self.encoder.fit(self.meta_frame[attribute])
        self.transform = transform
        self.use_one_channel = use_one_channel
        self.image_column = image_column

    def __len__(self):
        return len(self.meta_frame)

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
        sample = self.meta_frame.iloc[idx]

        img_name = os.path.join(self.root_dir, str(sample[self.image_column]))

        _, extension = os.path.splitext(img_name)
        loader_func = partial(loader_funcs[extension], use_one_channel=self.use_one_channel)
        image = loader_func(img_name)

        attribute_label = sample[self.attribute]
        attribute_label = self.encoder.transform([attribute_label])

        if self.transform:
            image = self.transform(image)
        return image, attribute_label[0]

class PatientChannelDataset(Dataset):
    def __init__(self, patient_muscle_csv_chart, root_dir, is_val, muscles_to_use=None, attribute='Diagnosis', transform=None):
        if muscles_to_use is None:
            muscles_to_use = ['D', 'B', 'FCR', 'R', 'G']
        self.muscle_channels = muscles_to_use
        meta_frame = pd.read_csv(patient_muscle_csv_chart)
        self.patients = parse_frame_into_patients(meta_frame)
        # merge left and right images of muscle
        self.patients = merge_left_right_channels(self.patients)
        # make pseudopatients for training purposes
        if not is_val:
            pp_list = []
            for patient in self.patients:
                pps = patient.make_pseudopatients(muscles=self.muscle_channels, method='sample_randomly')
                pp_list.extend(pps)
            self.patients = pp_list

        self.root_dir = root_dir
        self.attribute = attribute
        self.encoder = LabelEncoder()
        self.encoder.fit(meta_frame[attribute])
        self.transform = transform

    def get_all_atts(self, attribute):
        return [patient.attributes[attribute] for patient in self.patients]

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
        sample = self.patients[idx]
        sample_channels = sample.images_by_channel
        imgs = []
        for muscle in self.muscle_channels:
            img_list = sample_channels[muscle]
            # always take the first image, as there is always at least one
            name = str(img_list[0])

            img_name = name + '.jpg'
            img_name = os.path.join(self.root_dir, img_name)

            # load as PIL image
            with open(img_name, 'rb') as f:
                image = PIL.Image.open(f)
                image = image.convert('L')
                imgs.append(image)

        attribute_label = sample.attributes[self.attribute]
        attribute_label = self.encoder.transform([attribute_label])
        if self.transform:
            imgs = [self.transform(image) for image in imgs]

        # stack up the tensors at the end

        image = torch.stack(imgs).squeeze()
        return image, attribute_label[0]


loader_funcs = {'.png': load_img, '.jpg': load_img, '.dcm': load_dicom}

class PatientBagDataset(Dataset):
    def __init__(self, patient_list, root_dir,
                 use_pseudopatients, muscles_to_use=None,
                 attribute='Diagnosis', transform=None, use_one_channel=True):
        self.muscles_to_use = muscles_to_use
        self.patients = patient_list
        print(f'Loaded {len(self.patients)} patients.')
        # make pseudopatients for training purposes
        if use_pseudopatients:
            pp_list = []
            for patient in self.patients:
                pps = patient.make_pseudopatients(muscles=self.muscles_to_use, method='each_once')
                pp_list.extend(pps)
            self.patients = pp_list

        self.root_dir = root_dir
        self.attribute = attribute
        self.encoder = LabelEncoder()
        att_list = [x.attributes[attribute] for x in self.patients]
        self.encoder.fit(att_list)
        self.transform = transform
        self.use_one_channel = use_one_channel

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
        sample = self.patients[idx]
        sample_channels = sample.images_by_channel
        imgs = []
        if self.muscles_to_use is not None:
            gen = self.muscles_to_use
        else:
            gen = list(sample_channels.keys())

        for muscle in gen:
            img_list = sample_channels[muscle]
            # for now, always only take the first image, see whether we want to include
            # multiple images of the same muscle within the same bag
            img = img_list[0]
            name = str(img)

            img_name = os.path.join(self.root_dir, name)

            _, extension = os.path.splitext(img_name)
            loader_func = partial(loader_funcs[extension], use_one_channel=self.use_one_channel)
            image = loader_func(img_name)
            imgs.append(image)

            # one or three channels
            # mode = 'L' if self.use_one_channel else 'RGB'
            # # load as PIL image
            # with open(img_name, 'rb') as f:
            #     image = PIL.Image.open(f)
            #     image = image.convert(mode)
            #     imgs.append(image)

        attribute_label = sample.attributes[self.attribute]
        attribute_label = self.encoder.transform([attribute_label])
        if self.transform:
            imgs = [self.transform(image) for image in imgs]

        # stack up the tensors at the end
        image = torch.stack(imgs)
        return image, attribute_label[0]

    def __len__(self):
        return len(self.patients)


class Patient(object):

    def __init__(self, p_id, source, attributes, images_by_channel):
        self.p_id = p_id
        self.source = source
        # the attributes of this patient, e.g. age, diagnosis
        self.attributes = attributes
        # a mapping from channel_ids / muscles to image paths
        self.images_by_channel = images_by_channel

    def make_pseudopatients(self, muscles=None, method='each_once', n_limit=100):
        if not muscles:
            muscles = list(self.images_by_channel.keys())

        img_lists = [self.images_by_channel[muscle] for muscle in muscles]
        max_len = max([len(img_list) for img_list in img_lists])
        # make sure the lists all have the same length
        img_lists = [img_list if len(img_list) == max_len else repeat_to_length(img_list, max_len) for img_list in img_lists]

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

        pseudopatients = []
        for img_list in img_combs:
            img_list = [[x] for x in img_list]
            # zip up with muscle_labels
            images_by_channel = dict(zip(muscles,img_list))
            p = Patient(self.p_id, self.source, self.attributes, images_by_channel)
            pseudopatients.append(p)

        return pseudopatients

def parse_frame_into_patients(frame, data_source = 'albayda_github',
                              patient_id_name='PatientID', muscle_field_name='Muscle',
                              image_field_name='Image2D'):
    patient_ids = frame[patient_id_name].unique().tolist()
    # attributes that are not the same for every row within a patient
    non_uniform_atts = [patient_id_name, image_field_name, muscle_field_name]
    all_atts = set(frame.columns.to_list())
    uniform_attributes = all_atts - set(non_uniform_atts)
    patients = []
    for patient_id in patient_ids:
        patient_rows = frame[frame[patient_id_name] == patient_id]
        # grab the first row and read out the uniform_attributes
        first_row = patient_rows.iloc[0]
        attributes = {}
        for att in uniform_attributes:
            attributes[att] = first_row[att]

        # grab all the muscles for which we have a record from for this patient
        muscles = patient_rows[muscle_field_name].unique().tolist()
        images_by_muscle = {}
        for muscle in muscles:
            muscle_rows = patient_rows[patient_rows[muscle_field_name] == muscle]
            # get the ids for all images of this muscle
            image_ids = muscle_rows[image_field_name].unique().tolist()
            images_by_muscle[muscle] = image_ids

        p = Patient(p_id = patient_id, source=data_source, attributes=attributes, images_by_channel=images_by_muscle)
        patients.append(p)

    return patients

def count_channels(patients):
    channel_counts = []
    for patient in patients:
        all_channels = list(patient.images_by_channel.keys())
        channel_counts.extend(all_channels)
    return Counter(channel_counts)

def merge_left_right_channels(patients):
    new_patients = []
    for patient in patients:
        patient = deepcopy(patient)
        new_mapping = {}
        images_by_channel = patient.images_by_channel
        for k,v in images_by_channel.items():
            # chop off the R / L at the beginning
            new_key = k[1:]
            if new_key not in new_mapping:
                new_mapping[new_key] = []
            elem_list = new_mapping[new_key]
            elem_list.extend(v)
        patient.images_by_channel = new_mapping
        new_patients.append(patient)

    return new_patients

# path = '/home/klux/Thesis_2/data/myositis/im_muscle_chart.csv'
#
# frame = pd.read_csv(path)
# ps = parse_frame_into_patients(frame)
# new_ps = merge_left_right_channels(ps)
#
# counter = count_channels(new_ps)
# # D, B, FCR, R, G are present for all patients
# print(counter)
# pp_list = []
# for patient in new_ps:
#     pps = patient.make_pseudopatients(muscles=['D','B','FCR','R','G'], method='sample_randomly')
#     pp_list.extend(pps)
#
# print(len(pp_list))