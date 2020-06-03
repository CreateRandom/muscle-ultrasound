from functools import partial

from PIL import Image
from torch import is_tensor
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, label_binarize
from random import randint
import numpy as np
from tqdm import tqdm

from loading.img_utils import load_dicom, load_img, create_mask
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

loader_funcs = {'.png': load_img, '.jpg': load_img, '.dcm': load_dicom}

def select_latest(patient):
    return patient.select_latest()

class PatientBagDataset(Dataset):
    def __init__(self, patient_list, root_dir,
                 use_pseudopatients, muscles_to_use=None,
                 attribute='Diagnosis', transform=None, use_one_channel=True,
                 is_classification=True,
                 stack_images=True, use_mask=False, n_images_per_channel=1,
                 merge_left_right=False):
        self.muscles_to_use = muscles_to_use

        # a policy for record selection, TODO allow modification
        self.select_record_for_patient = select_latest
        self.patients = patient_list
        print(f'Loaded {len(self.patients)} patients.')
        for patient in self.patients:
            self.select_record_for_patient(patient)

        # make pseudopatients for training purposes
        if use_pseudopatients:
            pp_list = []
            for patient in tqdm(self.patients):
                pps = patient.make_pseudopatients(muscles=self.muscles_to_use, method='each_once')
                pp_list.extend(pps)
            self.patients = pp_list
            print(f'Pseudopatients yielded a total of {len(self.patients)} patients.')
        self.root_dir = root_dir
        self.attribute = attribute

        self.is_classification = is_classification
        # this needs to happen in case we treat this as a classification problem
        if self.is_classification:

            att_list = [x.attributes[self.attribute] for x in self.patients]
            self.classes = list(set(att_list))
            self.one_hot_encode_binary = False
            if len(self.classes) == 2 and self.one_hot_encode_binary:
                # the label binarizer does not properly one-hot encode binary attributes, so we'll cheat
                # add a dummy class and later remove it
                self.classes = self.classes + ['dummy_label']

        self.transform = transform
        self.use_one_channel = use_one_channel
        self.stack_images = stack_images
        self.use_mask = use_mask

        # these parameters control what's in the bag
        self.n_images_per_channel = n_images_per_channel
        # whether to bundle up images of the same muscle from left and right
        self.merge_left_right = merge_left_right
        if self.merge_left_right:
            self.grouper = ['Muscle']
        else:
            self.grouper = ['Muscle', 'Side']

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
        patient = self.patients[idx]
        sample = patient.get_selected_record()
        image_frame = sample.image_frame.copy()

        if self.muscles_to_use is not None:
            image_frame = image_frame['Muscle'].isin(self.muscles_to_use)

        def load_image(img):
            name = str(img)
            img_name = os.path.join(self.root_dir, name)
            raw_name, extension = os.path.splitext(img_name)
            loader_func = partial(loader_funcs[extension], use_one_channel=self.use_one_channel)
            image = loader_func(img_name)
            if self.use_mask:
                # also optionally try loading the mask
                try:
                    mat_file_path = raw_name + '.dcm.mat'
                    mask = create_mask(mat_file_path, image.size)
                    image2 = np.array(image)
                    mask = mask.transpose()
                    image2[~mask] = 0
                    image = Image.fromarray(image2)
                except:
                    print(f'Error loading mask for {name}')
            return image

        # retain n images for each muscle (and side) (n=1 --> the first image)
        image_frame = image_frame.groupby(self.grouper).head(self.n_images_per_channel)
        imgs = image_frame['ImagePath'].apply(load_image).to_list()

        if self.transform:
            imgs = [self.transform(image) for image in imgs]

        # stack up the tensors at the end
        if self.stack_images:
            imgs = torch.stack(imgs)
        if self.is_classification:
            label_to_return = self.get_classification_label(patient=patient)
        else:
            label_to_return = self.get_regression_label(patient=patient)

        return imgs, label_to_return

    def get_regression_label(self,patient):
        attribute_label = patient.attributes[self.attribute]
        return attribute_label

    def get_classification_label(self, patient):
        attribute_label = patient.attributes[self.attribute]
        transformed_label = label_binarize([attribute_label], classes=self.classes)
        if 'dummy_label' in self.classes:
            # drop the last column for the dummy class
            transformed_label = transformed_label[:, 0:-1]
        if self.one_hot_encode_binary:
            label_to_return = transformed_label[0]
        else:
            label_to_return = transformed_label[0][0]

        return label_to_return.astype(float)

    def __len__(self):
        return len(self.patients)

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
            self.record_to_use = np.argmax(dates,0)

    def make_pseudopatients(self, muscles=None, method='each_once', n_limit=100):
        record = self.records[self.record_to_use]
        pseudorecords = make_pseudorecords(record,muscles,method,n_limit)
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

    pseudorecords = []

    for img_list in img_combs:
        # img_list = [[x] for x in img_list]
        # zip up with muscle_labels
        # images_by_channel = dict(zip(muscles,img_list))
        reduced_image_frame = record.image_frame[record.image_frame['ImagePath'].isin(img_list)].copy()
        p = PatientRecord(record.r_id, reduced_image_frame, record.meta_info)
        pseudorecords.append(p)

    return pseudorecords

def parse_image_level_frame(frame):
    frame.rename(inplace=True,columns={'Image2D': 'ImagePath'})
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
