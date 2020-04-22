import os

import pandas as pd
from imgaug.augmenters import RandAugment
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import transforms, CenterCrop

from loading.datasets import parse_frame_into_patients, PatientBagDataset
from loading.img_utils import AugmentWrapper


def make_patient_bag_loader_myositis(csv_path, root_folder, attribute, transform, batch_size, is_val,
                                     use_one_channel):
    meta_frame = pd.read_csv(csv_path)
    # add image format
    meta_frame['Image2D'] = meta_frame['Image2D'].apply(lambda x: str(x) + '.jpg')
    patients = parse_frame_into_patients(meta_frame)
    ds = PatientBagDataset(patient_list=patients, root_dir=root_folder,
                           attribute=attribute, transform=transform, is_val=is_val,
                           muscles_to_use=None, use_one_channel=use_one_channel)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader


def make_transform_myositis(use_augment, use_one_channel=True, normalize=False):
    t_list = []
    # image size to rescale to, TODO unify
    r = transforms.Resize((224, 224))
    t_list.append(r)

    # data augmentation
    if use_augment:
        aug = AugmentWrapper(RandAugment())
        t_list.append(aug)

    t_list.append(transforms.ToTensor())

    if not use_one_channel and normalize:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        t_list.append(normalize)

    return transforms.Compose(t_list)


def make_myositis_loaders(train_path, val_path, img_folder, use_one_channel, normalize, attribute, batch_size):
    train_transform = make_transform_myositis(use_augment=False, use_one_channel=use_one_channel, normalize=normalize)
    train_loader = make_patient_bag_loader_myositis(train_path, img_folder,
                                                    attribute=attribute, transform=train_transform, batch_size=batch_size,
                                                    is_val=False, use_one_channel=use_one_channel)

    val_transform = make_transform_myositis(use_augment=False, use_one_channel=use_one_channel, normalize=normalize)
    val_loader = make_patient_bag_loader_myositis(val_path, img_folder,
                                                  attribute=attribute, transform=val_transform, batch_size=batch_size,
                                                  is_val=True, use_one_channel=use_one_channel)

    return train_loader, val_loader


def make_transform_umc(use_one_channel=True, normalize=False):
    t_list = []
    # standard image size (subject to change): 480 * 503 (narrower images are 480 * 335)

    # this does zero padding for smaller images
    center_crop = CenterCrop((480, 503))
    t_list.append(center_crop)

    # image size to rescale to
    # r = transforms.Resize((224, 224))
    # t_list.append(r)

    t_list.append(transforms.ToTensor())

    if not use_one_channel and normalize:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        t_list.append(normalize)

    return transforms.Compose(t_list)


def make_patient_bag_loader_umc(csv_path, root_folder, attribute, transform, batch_size, is_val,
                                use_one_channel):
    meta_frame = pd.read_csv(csv_path)
    # drop images with no ROI annotation
    meta_frame = meta_frame.dropna(subset=['min_h_roi'])
    print(f'Found a total of {len(meta_frame)} images.')

    # TODO make this more systematic
    muscles_to_use = ['Tibialis anterior', 'Biceps brachii', 'Gastrocnemius medial head', 'Flexor carpi radialis']
    meta_frame = meta_frame[meta_frame['Muscle'].isin(muscles_to_use)]
    print(f'Retained a total of {len(meta_frame)} images.')

    # merge folder and file path
    meta_frame['ImagePath'] = meta_frame.apply(lambda x: os.path.join(str(x['folder_name']), str(x['Image'])), axis=1)
    meta_frame.drop(inplace=True, columns=['folder_name', 'Image'])
    # todo allow storage of z-scores for each muscle rather than each patient
    patients = parse_frame_into_patients(meta_frame, data_source='umc',
                                         patient_id_name='pid', muscle_field_name='Muscle',
                                         image_field_name='ImagePath')

    ds = PatientBagDataset(patient_list=patients, root_dir=root_folder,
                           attribute=attribute, transform=transform, is_val=is_val,
                           muscles_to_use=None, use_one_channel=use_one_channel)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader


def make_umc_loaders(train_path, val_path, img_folder, use_one_channel, normalize, attribute, batch_size):
    train_transform = make_transform_umc(use_one_channel=use_one_channel, normalize=normalize)
    train_loader = make_patient_bag_loader_umc(train_path, img_folder,
                                                    attribute=attribute, transform=train_transform, batch_size=batch_size,
                                                    is_val=True, use_one_channel=use_one_channel)

    val_transform = make_transform_umc(use_one_channel=use_one_channel, normalize=normalize)
    val_loader = make_patient_bag_loader_umc(val_path, img_folder,
                                                  attribute=attribute, transform=val_transform, batch_size=batch_size,
                                                  is_val=True, use_one_channel=use_one_channel)

    return train_loader, val_loader