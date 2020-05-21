import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import transforms, CenterCrop

from loading.datasets import parse_frame_into_patients, PatientBagDataset, SingleImageDataset
from loading.img_utils import AugmentWrapper
from utils.utils import compute_normalization_parameters


def make_patient_bag_loader_myositis(csv_path, root_folder, attribute, transform, batch_size, use_pseudopatients,
                                     use_one_channel, pin_memory):
    meta_frame = pd.read_csv(csv_path)
    # add image format
    meta_frame['Image2D'] = meta_frame['Image2D'].apply(lambda x: str(x) + '.jpg')
    patients = parse_frame_into_patients(meta_frame)
    ds = PatientBagDataset(patient_list=patients, root_dir=root_folder,
                           attribute=attribute, transform=transform, use_pseudopatients=use_pseudopatients,
                           muscles_to_use=None, use_one_channel=use_one_channel)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory)
    return loader


def make_myositis_loaders(train_path, val_path, img_folder, use_one_channel, normalizer_name, attribute, batch_size,
                          use_pseudopatients=False, pin_memory=False, img_size=(475, 475)):
    train_transform = make_basic_transform(img_size, normalizer_name=normalizer_name)

    train_loader = make_patient_bag_loader_myositis(train_path, img_folder,
                                                    attribute=attribute, transform=train_transform,
                                                    batch_size=batch_size,
                                                    use_pseudopatients=use_pseudopatients,
                                                    use_one_channel=use_one_channel,
                                                    pin_memory=pin_memory)
    val_transform = make_basic_transform(img_size, normalizer_name=normalizer_name)

    val_loader = make_patient_bag_loader_myositis(val_path, img_folder,
                                                  attribute=attribute, transform=val_transform, batch_size=batch_size,
                                                  use_pseudopatients=False, use_one_channel=use_one_channel,
                                                  pin_memory=pin_memory)

    return train_loader, val_loader


normalizer_params = {'pretrained': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
                     'esaote6000': {'mean': [0.241], 'std': [0.141]}}


def make_basic_transform(img_size, normalizer_name=None):
    t_list = []

    # UMC: standard image size: 480 * 503 (narrower images are 480 * 335)
    # Myo: 475 * 475

    # this does zero padding for smaller images
    center_crop = CenterCrop(img_size)

    t_list.append(center_crop)

    # to tensor automatically scales between 0 and 1
    t_list.append(transforms.ToTensor())

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


def make_patient_bag_loader_umc(csv_path, root_folder, attribute, transform, batch_size, use_pseudopatients,
                                use_one_channel, pin_memory, muscles_to_use=None):
    meta_frame = preprocess_umc_patients(csv_path, muscles_to_use)

    # todo allow storage of z-scores for each muscle rather than each patient
    patients = parse_frame_into_patients(meta_frame, data_source='umc',
                                         patient_id_name='pid', muscle_field_name='Muscle',
                                         image_field_name='ImagePath')

    ds = PatientBagDataset(patient_list=patients, root_dir=root_folder,
                           attribute=attribute, transform=transform, use_pseudopatients=use_pseudopatients,
                           muscles_to_use=None, use_one_channel=use_one_channel)

    # TODO move to some other place
    def collate_function(batch):
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

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_function,
                        pin_memory=pin_memory)
    return loader


def select_muscles_to_use(csv_path, top_n=4):
    meta_frame = pd.read_csv(csv_path)
    # drop images with no ROI annotation
    meta_frame = meta_frame.dropna(subset=['min_h_roi'])

    x = meta_frame['Muscle'].value_counts().nlargest(top_n)
    muscles = set(x.index)
    return muscles


def make_umc_loaders(train_path, val_path, img_folder, use_one_channel, normalizer_name, attribute, batch_size,
                     use_pseudopatients=False, pin_memory=False, img_size=(475, 475)):
    train_transform = make_basic_transform(img_size=img_size, normalizer_name=normalizer_name)

    # select the most frequent 4 muscles from the training set
    muscles_to_use = select_muscles_to_use(train_path, top_n=4)

    # TODO allow comparison of different methods for using pseudopatients
    train_loader = make_patient_bag_loader_umc(train_path, img_folder,
                                               attribute=attribute, transform=train_transform, batch_size=batch_size,
                                               use_pseudopatients=use_pseudopatients, use_one_channel=use_one_channel,
                                               pin_memory=pin_memory, muscles_to_use=muscles_to_use)

    val_transform = make_basic_transform(img_size=img_size, normalizer_name=normalizer_name)

    val_loader = make_patient_bag_loader_umc(val_path, img_folder,
                                             attribute=attribute, transform=val_transform, batch_size=batch_size,
                                             use_pseudopatients=False, use_one_channel=use_one_channel,
                                             pin_memory=pin_memory,
                                             muscles_to_use=muscles_to_use)
    return train_loader, val_loader


def compute_empirical_mean_and_std(csv_path, root_folder, subsample=None, seed=None):
    transform = make_basic_transform((475, 475))
    meta_frame = preprocess_umc_patients(csv_path, muscles_to_use=None)
    if subsample:
        meta_frame = meta_frame.sample(n=subsample, random_state=seed)
    ds = SingleImageDataset(meta_frame, root_folder, attribute='Class', transform=transform, use_one_channel=True,
                            image_column='ImagePath')
    n_channels = 1
    mean, std = compute_normalization_parameters(ds, n_channels)
    return mean, std
