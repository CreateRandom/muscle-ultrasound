import json
import os
import pickle
import socket
from functools import partial
from inspect import signature
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import ImageEnhance
from scipy import stats
from scipy.stats import describe
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
from loading.datasets import PatientRecord, problem_kind, problem_legal_values, SingleImageDataset, make_att_specs
from loading.img_utils import load_image
from loading.loaders import get_data_for_spec, make_basic_transform
from loading.loading_utils import make_set_specs

from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report

from utils.utils import compute_normalization_parameters


class MuscleZScoreEncoder(object):

    def __init__(self, model_json_path) -> None:
        with open(model_json_path) as json_file:
            data = json.load(json_file)
        model_dicts = [self.parse_MATLAB_model(x) for x in data]
        names = [x['Muscle'] for x in data]
        self.model_mapping = dict(zip(names, model_dicts))

        # how to compute the predictors the models expect
        self.coeff_mapping = {'Age': lambda x: x['Age'], 'Lenght': lambda x: x['Height'],
                         'BMI': lambda x: x['BMI'], 'Age^2': lambda x: x['Age'] ** 2,
                         'Age^3': lambda x: x['Age'] ** 3, 'Weight': lambda x: x['Weight'],
                         'Sex_Male': lambda x: 1 if x['Sex'] == 'M' else 0,
                         'Dominance_Non-dominant': lambda x, y: int(x['Side'] != y)}

    @staticmethod
    def parse_MATLAB_model(model_json):
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
    # based on predci in CompactLinearModel.m
    def get_prediction_interval(x, coeff_cov, mse, crit=1.96):
        # add intercept term
        x = np.insert(x, 0, 1)
        var_pred = sum(np.matmul(x, coeff_cov) * x) + mse
        delta = np.sqrt(var_pred) * crit
        return delta

    def get_feature_rep(self,muscle, record, coeff_names):
        # build the feature representation
        x = []
        for coeff_name in coeff_names:
            #print(coeff_name)
            if coeff_name in self.coeff_mapping:
                extr_func = self.coeff_mapping[coeff_name]

                sig = signature(extr_func)
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
        muscle_name = muscle['Muscle']
        if muscle_name not in self.model_mapping:
            return np.nan
        coeff_names = self.model_mapping[muscle_name]['coefficient_names']
        # build the feature representation
        rep = self.get_feature_rep(muscle, record, coeff_names)

        # get the predicted value
        lr = self.model_mapping[muscle_name]['model']
        value_pred = lr.predict(rep.reshape(1, -1))
        #print(value_pred)
        # compute the z-score
        crit = stats.t.ppf(0.975, self.model_mapping[muscle_name]['dfe'])
        coeff_cov = self.model_mapping[muscle_name]['coefficient_cov']
        mse = self.model_mapping[muscle_name]['mse']
        margin_error = self.get_prediction_interval(rep, coeff_cov, mse, crit)

        #print(margin_error)
        SD = margin_error / crit
        #print(SD)
        z_score = (muscle['EI'] - value_pred) / SD
        return z_score[0]

    def encode_muscles(self, muscle_list, record):
        z_s = []
        for elem in muscle_list:
            z = self.encode_muscle(elem, record)
            z_s.append(z)
        return np.stack(z_s)

def compute_exceed_scores(Z_score_vector):
    to_return = {}
    to_return['NumberZscores'] = len(Z_score_vector)

    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    for t in thresholds:
        count = sum(Z_score_vector > t)
        name = 'Z_exceed' + str(t)
        to_return[name] = count
    return to_return

def get_mnt_path():
    current_host = socket.gethostname()
    if current_host == 'pop-os':
        mnt_path = '/mnt/chansey/'
    else:
        mnt_path = '/mnt/netcache/diag/'
    return mnt_path

def get_default_set_spec_dict():
    mnt_path = get_mnt_path()
    print(f'Using mount_path: {mnt_path}')

    umc_data_path = os.path.join(mnt_path, 'klaus/data/devices/')
    umc_img_root = os.path.join(mnt_path, 'klaus/total_patients/')
    jhu_data_path = os.path.join(mnt_path, 'klaus/myositis/')
    jhu_img_root = os.path.join(mnt_path, 'klaus/myositis/processed_imgs')
    set_spec_dict = make_set_specs(umc_data_path, umc_img_root, jhu_data_path, jhu_img_root)
    return set_spec_dict

def run_dummy_baseline(set_name, attribute):
    set_spec_dict = get_default_set_spec_dict()
    set_spec = set_spec_dict[set_name]
    filter_attribute = 'Class_sample' if attribute == 'Class' else None
    class_values = problem_legal_values[attribute] if attribute in problem_legal_values else None
    patients = get_data_for_spec(set_spec, loader_type='bag', attribute_to_filter=attribute,
                                 legal_attribute_values=class_values,
                                 muscles_to_use=None, boolean_subset_attribute=filter_attribute)

    y_true = [patient.attributes[attribute] for patient in patients]

    kind = problem_kind[attribute]

    if kind == 'regression':
        d = DummyRegressor(strategy='mean')
        scorer = mean_absolute_error

    else:
        d = DummyClassifier(strategy='most_frequent')
        scorer = classification_report
    d.fit([0] * len(y_true), y_true)
    train_preds = d.predict([0] * len(y_true))

    print(scorer(y_true, train_preds))

def compute_EIZ_from_scratch(record, root_path, z_score_encoder):
    
    load_func = partial(load_image, root_dir=root_path, use_one_channel=True,
                        use_mask=True)

    def compute_EI_manually(img_path):
        pil_image = load_func(img_path)

        np_array = np.asarray(pil_image)
        ei = np.mean(np_array[np_array > 0])
        return ei

    record.image_frame['EI_manual'] = record.image_frame['ImagePath'].apply(compute_EI_manually).to_list()

    manual_EI = record.image_frame.groupby(['Muscle', 'Side']).mean()['EI_manual']
    manual_EI = round(manual_EI).astype(int)
    manual_muscles = manual_EI.index.get_level_values(0).values
    manual_sides = manual_EI.index.get_level_values(1).values

    muscles_to_encode = [{'Muscle': v1, 'Side': v2, 'EI': v3} for v1, v2, v3 in
                                zip(manual_muscles, manual_sides, manual_EI)]

    EIZ = z_score_encoder.encode_muscles(muscles_to_encode, record.meta_info)

    return EIZ

handedness_mapping = {'L' : 'Links', 'R': 'Rechts'}

def obtain_zscores(EIs, record, z_score_encoder):
    handedness = [handedness_mapping[x] for x in record.meta_info['Sides']]

    muscles_to_encode = [{'Muscle': v1, 'Side': v2, 'EI': v3} for v1, v2, v3 in
                                zip(record.meta_info['Muscles_list'], handedness, EIs)]

    EIZ = z_score_encoder.encode_muscles(muscles_to_encode, record.meta_info)

    return EIZ

def adjust_EI_with_factor(record, factor):
    mapped_EI = np.array(record.meta_info['EIs']) * factor
    return mapped_EI

def adjust_EI_with_regression(record, lr_model):
    mapped_EI = lr_model.predict(np.array(record.meta_info['EIs']).reshape(-1, 1)).tolist()
    return mapped_EI

def obtain_scores_and_preds(EIZ, record, name):

    new_exceed_scores = compute_exceed_scores(EIZ)
    new_exceed_scores['Age'] = record.meta_info['Age']

    new_pred = predict_rule_based(new_exceed_scores)
    new_exceed_scores['pred'] = new_pred
    new_exceed_scores.pop('Age')
    new_exceed_scores['method'] = name
    new_exceed_scores['pid'] = record.meta_info['pid']
    return new_exceed_scores


def run_rule_based_baseline(set_name):
    set_spec_dict = get_default_set_spec_dict()
    set_spec = set_spec_dict[set_name]
    patients = get_data_for_spec(set_spec, loader_type='bag', attribute_to_filter='Class',
                                 legal_attribute_values=problem_legal_values['Class'],
                                 muscles_to_use=None, boolean_subset_attribute='Class_sample')

    # patients = get_data_for_spec(set_spec, loader_type='bag', attribute='Sex',
    #                               muscles_to_use=None)
    all_preds = []
    y_true = []
    all_zscores = []
    z_score_encoder = MuscleZScoreEncoder('data/MUS_EI_models.json')
    with open('linear_regression_philips_to_esoate.pkl', 'rb') as f:
        lr_model = pickle.load(f)

    for patient in patients:
        patient.select_latest()
        record = patient.get_selected_record()
        # predictions based on the original z-scores
        all_preds.append(obtain_scores_and_preds(np.array(record.meta_info['EIZ']), record, 'original'))
        all_zscores.append({'method': 'original' ,'data': np.array(record.meta_info['EIZ'])})

        # prediction based on the z-score encoder for ESOATE
        EIZ_from_EI = obtain_zscores(np.array(record.meta_info['EIs']), record, z_score_encoder)
        all_zscores.append({'method': 'unadjusted_ei_esoate_eiz', 'data' : EIZ_from_EI})
        all_preds.append(obtain_scores_and_preds(EIZ_from_EI, record,'unadjusted_ei_esoate_eiz'))

        # prediction base
        EI_regression = adjust_EI_with_regression(record,lr_model)
        EIZ_regression = obtain_zscores(EI_regression, record, z_score_encoder)
        all_zscores.append({'method': 'regression_ei_esoate_eiz', 'data': EIZ_regression})
        all_preds.append(obtain_scores_and_preds(EIZ_regression, record, 'regression_ei_esoate_eiz'))

        EI_adjustment = adjust_EI_with_factor(record, 1.4364508393285371)
        EIZ_adjustment = obtain_zscores(EI_adjustment, record, z_score_encoder)
        all_zscores.append({'method': 'adjusted_ei_esoate_eiz', 'data': EIZ_adjustment})
        all_preds.append(obtain_scores_and_preds(EIZ_adjustment,record, 'adjusted_ei_esoate_eiz'))

      #  EIZ_image = compute_EIZ_from_scratch(record, set_spec.img_root_path, z_score_encoder=z_score_encoder)
      #  pred_dict_image = obtain_scores_and_preds(EIZ_image,record)


        #  EIZ = np.array(record.meta_info['EIZ'])

        #     nan_inds = np.argwhere(np.isnan(EIZ))
        #     EIZ_corr = np.delete(EIZ, nan_inds)
        #     EIZ_new_corr = np.delete(EIZ_new, nan_inds)
        #     r, _ = stats.pearsonr(EIZ_corr, EIZ_new_corr)
        #    print(record.meta_info['RecordingDate'])
        #    print(r)

        #     plt.plot(EIZ, EIZ_new, 'r*')
        #     plt.show()

        y_true.append(patient.attributes['Class'])

    pred_frame = pd.DataFrame(all_preds)
    z_score_frame = pd.DataFrame(all_zscores)

    for method, group_frame in z_score_frame.groupby('method'):
        all_scores = np.concatenate(group_frame.data.values)
        nan_inds = np.argwhere(np.isnan(all_scores))
        print(method)
        print(describe(np.delete(all_scores, nan_inds)))


    p = np.array(patients)

    error_frames = {}
    # score all the methods
    for method, group_frame in pred_frame.groupby('method'):
        print(method)
        preds = group_frame['pred']
        y_pred_recall_biased = [elem if elem != 'unknown or uncertain' else 'NMD' for elem in preds]
        y_pred_precision_biased = [elem if elem != 'unknown or uncertain' else 'no NMD' for elem in preds]
        print('Focus on recall')
        print(classification_report(y_true, y_pred_recall_biased))
        print('Focus on precision')
        print(classification_report(y_true, y_pred_precision_biased))

        # error analysis
        fps = np.where((np.array(y_true) == 'no NMD') & (np.array(y_pred_recall_biased) == 'NMD'))
        fns = np.where((np.array(y_true) == 'NMD') & (np.array(y_pred_recall_biased) == 'no NMD'))

        info_dicts = []
        x = p[fps]
        for patient in x:
            info = patient.get_selected_record().meta_info
            info['error_type'] = 'fp'
            info_dicts.append(info)
        x = p[fns]
        for patient in x:
            info = patient.get_selected_record().meta_info
            info['error_type'] = 'fn'
            info_dicts.append(info)

        error_frame = pd.DataFrame(info_dicts)
        error_frames[method] = error_frame



def predict_rule_based(dict_with_exceed):
    if np.isnan(dict_with_exceed['NumberZscores']):
        return np.nan
    # age below 65
    if dict_with_exceed['Age'] < 65:
        # fewer than six z-scores recorded
        if dict_with_exceed['NumberZscores'] < 6:
            # one muscle exceeds 3.5
            if dict_with_exceed['Z_exceed3.5'] >= 1:
                return 'NMD'
            # two muscles exceed 2.5
            if dict_with_exceed['Z_exceed2.5'] >= 2:
                return 'NMD'
            # three exceed 1.5
            if dict_with_exceed['Z_exceed1.5'] >= 3:
                return 'NMD'
            # one muscle exceeds 2.0
            if dict_with_exceed['Z_exceed2.0'] >= 1:
                return 'unknown or uncertain'
            # two muscles exceeds 1.5
            if dict_with_exceed['Z_exceed1.5'] >= 2:
                return 'unknown or uncertain'
            # three exceed 1.0
            if dict_with_exceed['Z_exceed1.0'] >= 3:
                return 'unknown or uncertain'
            # basecase
            return 'no NMD'

        # six or more z-scores
        else:
            # one muscle exceeds 3.5
            if dict_with_exceed['Z_exceed3.5'] >= 1:
                return 'NMD'
            # two muscles exceed 2.5
            if dict_with_exceed['Z_exceed2.5'] >= 2:
                return 'NMD'
            # three exceed 1.5
            if dict_with_exceed['Z_exceed1.5'] >= 3:
                return 'NMD'
            # one muscle exceeds 2.0
            if dict_with_exceed['Z_exceed2.0'] >= 1:
                return 'unknown or uncertain'
            # basecase
            return 'no NMD'

    # age above 65
    else:
        # fewer than six z-scores
        if dict_with_exceed['NumberZscores'] < 6:
            # one muscle exceeds 3.5
            if dict_with_exceed['Z_exceed3.5'] >= 1:
                return 'NMD'
            # two muscles exceed 2.5
            if dict_with_exceed['Z_exceed2.5'] >= 2:
                return 'NMD'
            # three exceed 1.5
            if dict_with_exceed['Z_exceed1.5'] >= 3:
                return 'NMD'
            # *one muscle exceeds 2.5*
            if dict_with_exceed['Z_exceed2.5'] >= 1:
                return 'unknown or uncertain'
            # basecase
            return 'no NMD'

        # six or more z-scores
        else:
            # one muscle exceeds 3.5
            if dict_with_exceed['Z_exceed3.5'] >= 1:
                return 'NMD'
            # two muscles exceed 2.5
            if dict_with_exceed['Z_exceed2.5'] >= 2:
                return 'NMD'
            # *three exceed 2.0*
            if dict_with_exceed['Z_exceed2.0'] >= 3:
                return 'NMD'
            # *one muscle exceeds 3.0*
            if dict_with_exceed['Z_exceed3.0'] >= 1:
                return 'unknown or uncertain'
            # basecase
            return 'no NMD'

def analyze_multi_device_patients():
    set_spec_dict = get_default_set_spec_dict()
    all_patients = []
    for key, value in set_spec_dict.items():
        if key.startswith('Multiple'):
            patients = get_data_for_spec(value, loader_type='bag', attribute_to_filter=None,
                                         muscles_to_use=None)
            all_patients.extend(patients)

    match_table = []
    for patient in all_patients:
        p, e = get_matched_records(patient)
        if not p:
            continue
        match_table.append(align_records(p, e))

    total = pd.concat(match_table)


    total.to_pickle('multi_patients_aligned.pkl')

def get_record_frame(patient):
    info = []
    for i, record in enumerate(patient.records):
        info.append({**record.meta_info, **{'i': i}})

    record_frame = pd.DataFrame(info)

    return record_frame

def get_matched_records(patient):
    record_frame = get_record_frame(patient)

    if not 'ESAOTE_6100' in record_frame['DeviceInfo'].values or not 'Philips Medical Systems_iU22' in record_frame['DeviceInfo'].values:
        return None, None

    esaote_records = record_frame[record_frame['DeviceInfo'] == 'ESAOTE_6100'].sort_values('RecordingDate', ascending=True)
    esaote_record = esaote_records.iloc[0]
    esaote_record = patient.records[esaote_record['i']]
    philips_records = record_frame[record_frame['DeviceInfo'] == 'Philips Medical Systems_iU22'].sort_values('RecordingDate',
                                                                                           ascending=False)
    philips_record = philips_records.iloc[0]
    philips_record = patient.records[philips_record['i']]

    return philips_record, esaote_record

def get_muscle_frame(record):
    muscles_to_encode = [{'Muscle': v1, 'Side': v2, 'EI': v3, 'EIZ': v4} for v1, v2, v3, v4 in
                         zip(record.meta_info['Muscles_list'], record.meta_info['Sides'], record.meta_info['EIs'], record.meta_info['EIZ'])]

    return pd.DataFrame(muscles_to_encode)

def align_records(record_a, record_b):
    date_diff = record_b.meta_info['RecordingDate'] - record_a.meta_info['RecordingDate']
    muscle_frame_a = get_muscle_frame(record_a)

    muscle_frame_b = get_muscle_frame(record_b)

    x = muscle_frame_a.merge(muscle_frame_b, on=['Muscle', 'Side'], suffixes=('_p', '_e'))
    x = x.dropna(subset=['EI_p', 'EI_e'])
    x['DateDiff'] = date_diff
    return x


def compute_brightness_diff():

    att_spec_dict = make_att_specs()
    set_spec_dict = get_default_set_spec_dict()
    esoate_spec = set_spec_dict['Philips_iU22_train']
    esoate_images = get_data_for_spec(esoate_spec, loader_type='image', dropna_values=False)
    esoate_images = esoate_images[0:100]

    transform = make_basic_transform('Philips_iU22', limit_image_size=False, to_tensor=True)

    ds = SingleImageDataset(image_frame=esoate_images, root_dir=esoate_spec.img_root_path, attribute_specs=[att_spec_dict['Sex']],
                            return_attribute_dict=False, transform=transform,
                            use_one_channel=True)


    mean, std  = compute_normalization_parameters(ds, 1)

    print(mean)
    print(std)


if __name__ == '__main__':
 #   compute_brightness_diff()
 #   analyze_multi_device_patients()
    run_rule_based_baseline('Philips_iU22_val')
    run_dummy_baseline('ESAOTE_6100_val', 'Sex')