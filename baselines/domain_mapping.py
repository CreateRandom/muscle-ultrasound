import pickle
from functools import partial

from baselines.brightness import get_brightness_factor
from baselines.zscores import MuscleZScoreEncoder, get_brightness_adjusted_z_scores, get_regression_adjusted_z_scores, \
    get_recomputed_z_scores, compute_EIZ_from_scratch, get_original_scores

# location of stored linear regression models trained using the EI_transformation notebook
lin_reg_storage = {'Philips_iU22': {'ESAOTE_6100': 'final_models/linear_regression_esaote_to_philips.pkl'},
           'ESAOTE_6100': {'Philips_iU22': 'final_models/linear_regression_philips_to_esoate.pkl'}}

# image mappings

# N.B.
# A is Esaote, so fake A is the mapped Philips and fake B is mapped Esaote
# for these experiments, we always want to use the mapped target

# mapped image folders obtained from the Standard cyclegan
standard_cyclegan = {'Philips_iU22': {'ESAOTE_6100': '/media/klux/Elements/standard_cyclegan/fakeA'},
                    'ESAOTE_6100': {'Philips_iU22': '/media/klux/Elements/standard_cyclegan/fakeB'}}

# mapped images folders obtained from CyCADA with semantic loss
cycada_sem_paths = {'Philips_iU22': {'ESAOTE_6100': '/home/klux/Thesis_2/images_cycada/mappedA'},
                    'ESAOTE_6100': {'Philips_iU22': '/home/klux/Thesis_2/images_cycada/mappedB'}}

# mapped images folders obtained from CyCADA without semantic loss
cycada_no_sem_paths = {'Philips_iU22': {'ESAOTE_6100': '/home/klux/Thesis_2/images_cycada_no_sem/mappedA'},
                    'ESAOTE_6100': {'Philips_iU22': '/home/klux/Thesis_2/images_cycada_no_sem/mappedB'}}

# different image mapping conditions that can be compared
image_mappings = {'standard_cyclegan': standard_cyclegan,
                  'cycada_semantic': cycada_sem_paths,
                  'cycada_no_semantic': cycada_no_sem_paths}

def get_z_score_encoder(source_set):
    # we currently have only one z-score encoder, the values for Philips were not recorded in an accessible way
    if source_set == 'ESAOTE_6100':
        return MuscleZScoreEncoder('final_models/MUS_EI_models.json')


def get_lr_model(source_set, target_set):
    model_path = lin_reg_storage[source_set][target_set]
    with open(model_path, 'rb') as f:
        lr_model = pickle.load(f)
    return lr_model


def get_mapped_path(source_set, target_set, method='standard_cyclegan'):
    mapped_path_dict = image_mappings[method]
    return mapped_path_dict[source_set][target_set]

def get_domain_mapping_method(type, source_set, target_set, z_score_encoder=None):
    '''
    A helper function to help create the correct domain mapping method.
    :param type: What type of adjustment to perform.
    :param source_set: The source domain.
    :param target_set: The target domain.
    :param z_score_encoder: A z-score encoder
    :return: A function to perform the adjustment of the patient records.
    '''
    ei_extraction_method = None
    # brightness-based mapping
    if type == 'brightness':
        factor = get_brightness_factor(source_set.device, target_set.device)
        ei_extraction_method = partial(get_brightness_adjusted_z_scores, z_score_encoder=z_score_encoder, factor=factor)
    # linear regression mapping based on multi-device patients
    if type == 'regression':
        lr_model = get_lr_model(source_set.device,target_set.device)
        ei_extraction_method = partial(get_regression_adjusted_z_scores, z_score_encoder=z_score_encoder,
                                           regression_model=lr_model)
    # simply recompute EIZ scores from EI scores, e.g. EI scores from Philips with Esoate EIZ encoder to measure
    # performance gap
    if type == 'recompute':
        ei_extraction_method = partial(get_recomputed_z_scores, z_score_encoder=z_score_encoder)
    # image mapping using stored mapped images.
    # It proved impractical to directly integrate the image mapping neural network, so instead
    # all images are mapped beforehand a stored on a path, from which they then can be loaded by
    # means of the below methods.
    if type == 'mapped_images':
        mapped_path = get_mapped_path(source_set.device, target_set.device)
        mask_root_path = target_set.img_root_path
        ei_extraction_method = partial(compute_EIZ_from_scratch, z_score_encoder=z_score_encoder, root_path=mapped_path,
                                       strip_folders=True, mask_root_path=mask_root_path)
    # retrieve the original scores
    if type == 'original':
        ei_extraction_method = get_original_scores

    if not ei_extraction_method:
        raise ValueError(f'Unknown method for adjustment: {type}')

    return ei_extraction_method