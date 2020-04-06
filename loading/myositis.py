from collections import Counter
from copy import deepcopy

from torch import is_tensor
import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import PIL.Image
from random import randint

from utils.utils import repeat_to_length
import torch

class MyositisDataset(Dataset):

    def __init__(self, csv_file, root_dir, attribute='Diagnosis', transform=None,
                 use_one_channel=False):
        """
        Args:
            csv_file (string): Path to the csv file with meta info.
            root_dir (string): Directory with all the images.
            attribute (string): The attribute to output. Default: Diagnosis.
        """
        self.meta_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.attribute = attribute
        self.encoder = LabelEncoder()
        self.encoder.fit(self.meta_frame[attribute])
        self.transform = transform
        self.use_one_channel = use_one_channel

    def __len__(self):
        return len(self.meta_frame)

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
        sample = self.meta_frame.iloc[idx]
        img_name = str(str(sample['Image2D']) + '.jpg')
        img_name = os.path.join(self.root_dir, img_name)

        # load as PIL image
        with open(img_name, 'rb') as f:
            image = PIL.Image.open(f)
            # Luminance if only one channel, else RGB
            type = 'L' if self.use_one_channel else 'RGB'
            image = image.convert(type)

        attribute_label = sample[self.attribute]
        attribute_label = self.encoder.transform([attribute_label])

        if self.transform:
            image = self.transform(image)
        return image, attribute_label[0]

class PatientDataset(Dataset):
    def __init__(self, patient_muscle_csv_chart, root_dir, is_val, muscle_channels=None, attribute='Diagnosis', transform=None):
        if muscle_channels is None:
            muscle_channels = ['D', 'B', 'FCR', 'R', 'G']
        self.muscle_channels = muscle_channels
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



class Patient(object):

    def __init__(self, p_id, source, attributes, images_by_channel):
        self.p_id = p_id
        self.source = source
        # the attributes of this patient, e.g. age, diagnosis
        self.attributes = attributes
        # a mapping from channel_ids / muscles to image paths
        self.images_by_channel = images_by_channel

    def make_pseudopatients(self, muscles, method='each_once', n_limit=100):
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

def parse_frame_into_patients(frame):
    patient_ids = frame['PatientID'].unique().tolist()
    # attributes that are not the same for every row within a patient
    non_uniform_atts = ['PatientID', 'Image2D', 'Muscle']
    all_atts = set(frame.columns.to_list())
    uniform_attributes =  all_atts - set(non_uniform_atts)
    patients = []
    for patient_id in patient_ids:
        patient_rows = frame[frame['PatientID'] == patient_id]
        # grab the first row and read out the uniform_attributes
        first_row = patient_rows.iloc[0]
        attributes = {}
        for att in uniform_attributes:
            attributes[att] = first_row[att]

        # grab all the muscles for which we have a record from for this patient
        muscles = patient_rows['Muscle'].unique().tolist()
        images_by_muscle = {}
        for muscle in muscles:
            muscle_rows = patient_rows[patient_rows['Muscle'] == muscle]
            # get the ids for all images of this muscle
            image_ids = muscle_rows['Image2D'].unique().tolist()
            images_by_muscle[muscle] = image_ids

        p = Patient(p_id = patient_id, source='albayda_github', attributes=attributes, images_by_channel=images_by_muscle)
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