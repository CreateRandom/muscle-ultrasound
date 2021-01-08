import json
import os
from inspect import signature

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

from loading.img_utils import load_image


class MuscleZScoreEncoder(object):
    """
    An encoder that converts raw brightness values into
    Z scores by using a regression model trained as a MATLAB json file.
    """

    def __init__(self, model_json_path) -> None:
        with open(model_json_path) as json_file:
            data = json.load(json_file)
        # there is one model per muscle, store them in a dictionary
        model_dicts = [self.parse_MATLAB_model(x) for x in data]
        names = [x['Muscle'] for x in data]
        # maps from muscle names to z score models
        self.model_mapping = dict(zip(names, model_dicts))

        # how to compute the predictors the models expect
        self.coeff_mapping = {'Age': lambda x: x['Age'], 'Lenght': lambda x: x['Height'],
                         'BMI': lambda x: x['BMI'], 'Age^2': lambda x: x['Age'] ** 2,
                         'Age^3': lambda x: x['Age'] ** 3, 'Weight': lambda x: x['Weight'],
                         'Sex_Male': lambda x: 1 if x['Sex'] == 'M' else 0,
                         'Dominance_Non-dominant': lambda x, y: int(x['Side'] != y)}

    @staticmethod
    def parse_MATLAB_model(model_json):
        """
        Reading in a stored matlab z score model for use in python
        :param model_json: dict read from the JSON saved by MATLAB
        :return: a dict with the necessary information
        """
        # muscle = model_json['Muscle']
        rmse = model_json['RMSE']
        mse = model_json['MSE']
        dfe = model_json['DFE']
        coefficient_cov = model_json['CoeffCov']
        intercept = model_json['Coefficients'][0]
        names = model_json['CoefficientNames'][1:]
        coeff = model_json['Coefficients'][1:]

        lr = LinearRegression()
        lr.coef_ = np.array(coeff)
        lr.intercept_ = intercept

        return {'model': lr, 'coefficient_names': names, 'rmse': rmse, 'dfe': dfe,
                'mse': mse, 'coefficient_cov': np.array(coefficient_cov)}

    @staticmethod
    def get_prediction_interval(x, coeff_cov, mse, crit=1.96):
        """
        This method emulates predci in CompactLinearModel.m to obtain a prediction interval
        :param x: mean estimate observed
        :param coeff_cov: Covariance coefficients
        :param mse: mean squared error
        :param crit: critical interval
        :return: the prediction interval around x
        """
        # add intercept term
        x = np.insert(x, 0, 1)
        var_pred = sum(np.matmul(x, coeff_cov) * x) + mse
        delta = np.sqrt(var_pred) * crit
        return delta

    def get_feature_rep(self,muscle, record, coeff_names):
        """
        Encode the muscle of the record using the coefficient names specified as features.
        :param muscle: muscle information in dictionary form
        :param record: a patient record in dictionary form
        :param coeff_names: the names of the model coefficients to be extracted
        :return: a vector that contains the necessary EI scores of the muscle in question
        """
        # build the feature representation
        x = []
        for coeff_name in coeff_names:
            #print(coeff_name)
            if coeff_name in self.coeff_mapping:
                # get the function that extracts this coefficient from the record
                extr_func = self.coeff_mapping[coeff_name]
                # some functions need extra information, find out about this by using their signature
                sig = signature(extr_func)
                # find how many parameters the function needs
                n_param = len(sig.parameters)
                if n_param == 1:
                    new_val = extr_func(record)
                else:
                    new_val = extr_func(record, muscle['Side'])
                # ensure not nan
                if np.isnan(new_val):
                    new_val = 0
               #     raise Warning(f'Missing value {coeff_name} for record {record}')
                x.append(new_val)
            else:
                raise ValueError(f'Missing {coeff_name}')
        rep = np.stack(x)
        return rep

    def encode_muscle(self, muscle, record):
        '''
        Obtain the z-score for a given muscle on a given record
        :param muscle: a muscle dict
        :param record: a record dict
        :return: The z-score obtained using the model
        '''
        muscle_name = muscle['Muscle']
        if muscle_name not in self.model_mapping:
            return np.nan
        coeff_names = self.model_mapping[muscle_name]['coefficient_names']
        # build the feature representation
        rep = self.get_feature_rep(muscle, record, coeff_names)

        # get the predicted value
        lr = self.model_mapping[muscle_name]['model']
        value_pred = lr.predict(rep.reshape(1, -1))
        # compute the z-score
        # this is the Python version of the MATLAB code I was provided
        crit = stats.t.ppf(0.975, self.model_mapping[muscle_name]['dfe'])
        coeff_cov = self.model_mapping[muscle_name]['coefficient_cov']
        mse = self.model_mapping[muscle_name]['mse']
        margin_error = self.get_prediction_interval(rep, coeff_cov, mse, crit)

        SD = margin_error / crit
        z_score = (muscle['EI'] - value_pred) / SD
        return z_score[0]

    def encode_muscles(self, muscle_list, record):
        '''
        Batch encode all muscles in the list
        :param muscle_list: a list of muscles
        :param record: a record in dictionary form
        :return: a vector with z scores
        '''
        z_s = []
        for elem in muscle_list:
            z = self.encode_muscle(elem, record)
            z_s.append(z)
        return np.stack(z_s)


def compute_EIZ_from_scratch(record, z_score_encoder, root_path, strip_folders=False,
                             mask_root_path=None, factor=None):
    '''
    A method for computing EI (z-scores) from the underlying image directly, rather than from
    pre-computed EI scores.
    :param record: a patient record
    :param z_score_encoder: a model for mapping from EI to EIZ
    :param root_path: the folder that contains images
    :param strip_folders: for searching for the location of masks relative to images
    :param mask_root_path: a separate folder that contains image masks
    :param factor: a brightness adjustment factor applied to EI scores
    :return: a dictionary with EI and EIZ scores
    '''
    if not mask_root_path:
        mask_root_path = root_path

    def compute_EI_manually(row):

        image_location = row['ImageFile']
        mask_location = row['ImageFile'] if not strip_folders else row['ImagePath']

        mask_location = os.path.join(mask_root_path, mask_location)
        try:
            pil_image = load_image(image_location,
                                   root_dir=root_path, use_one_channel=True, use_mask=True, original_image_location= mask_location)

            np_array = np.asarray(pil_image)
            ei = np.mean(np_array[np_array > 0])
        except:
            print(f'Missing {row}')
            ei = np.nan
        return ei

    if strip_folders:
        # only use the filename
        record.image_frame['ImageFile'] = record.image_frame['ImagePath'].apply(lambda x: os.path.basename(x))
    else:
        # use the full_path
        record.image_frame['ImageFile'] = record.image_frame['ImagePath']
    record.image_frame['EI_manual'] = record.image_frame.swifter.apply(compute_EI_manually,axis=1).to_list()

    manual_EI = record.image_frame.groupby(['Muscle', 'Side']).mean()['EI_manual']
    manual_EI = round(manual_EI).astype(int)
    manual_muscles = manual_EI.index.get_level_values(0).values
    manual_sides = manual_EI.index.get_level_values(1).values

    muscles_to_encode = [{'Muscle': v1, 'Side': v2, 'EI': v3} for v1, v2, v3 in
                                zip(manual_muscles, manual_sides, manual_EI)]
    return_dict = {}
    return_dict['EI'] = manual_EI
    if factor:
        return_dict['EI'] = return_dict['EI'] * factor
    if z_score_encoder:
        EIZ = z_score_encoder.encode_muscles(muscles_to_encode, record.meta_info)
        return_dict['EIZ'] = EIZ

    return return_dict


def obtain_zscores(EIs, record, z_score_encoder):
    '''
    Given an array of EI scores, a patient record and an encoder, get the EIZ scores.
    :param EIs: an array of EI scores
    :param record: a patient record
    :param z_score_encoder: a ZScoreEncoder object
    :return: an array of EIZ scores
    '''
    handedness_mapping = {'L': 'Links', 'R': 'Rechts'}

    handedness = [handedness_mapping[x] for x in record.meta_info['Sides']]

    muscles_to_encode = [{'Muscle': v1, 'Side': v2, 'EI': v3} for v1, v2, v3 in
                                zip(record.meta_info['Muscles_list'], handedness, EIs)]

    EIZ = z_score_encoder.encode_muscles(muscles_to_encode, record.meta_info)

    return EIZ


def adjust_EI_with_factor(record, factor):
    """ Adjust EI by a fixed factor. """
    mapped_EI = np.array(record.meta_info['EIs']) * factor
    return mapped_EI


def adjust_EI_with_regression(record, lr_model):
    """ Adjust EI with a regression model trained in the EI_transformation notebook."""
    mapped_EI = lr_model.predict(np.array(record.meta_info['EIs']).reshape(-1, 1)).tolist()
    return np.array(mapped_EI)


def get_original_scores(record):
    """Returns the original EI and EIZ scores"""
    return {'EI': np.array(record.meta_info['EIs']), 'EIZ': np.array(record.meta_info['EIZ'])}


def get_recomputed_z_scores(record, z_score_encoder):
    """ Recomputes the Z scores from the stored EI scores, using the provided encoder."""
    return_dict = {}
    EIs = np.array(record.meta_info['EIs'])
    return_dict['EI'] = EIs
    if z_score_encoder:
        EIZ = obtain_zscores(EIs, record, z_score_encoder)
        return_dict['EIZ'] = EIZ
    return return_dict


def get_regression_adjusted_z_scores(record,z_score_encoder, regression_model):
    """Adjusts the EI scores using a regression model (trained in EI_transformation notebook)
    and then transforms them into EIZ scores."""
    return_dict = {}
    EI_regression = adjust_EI_with_regression(record, regression_model)
    return_dict['EI'] = EI_regression
    if z_score_encoder:
        EIZ_regression = obtain_zscores(EI_regression, record, z_score_encoder)
        return_dict['EIZ'] = EIZ_regression
    return return_dict


def get_brightness_adjusted_z_scores(record, z_score_encoder, factor):
    """Adjusts EI scores using a fixed brightness factor estimated from data
    and then transforms them into EIZ scores."""
    return_dict = {}
    EI_adjustment = adjust_EI_with_factor(record, factor)
    return_dict['EI'] = EI_adjustment
    if z_score_encoder:
        EIZ_regression = obtain_zscores(EI_adjustment, record, z_score_encoder)
        return_dict['EIZ'] = EIZ_regression
    return return_dict