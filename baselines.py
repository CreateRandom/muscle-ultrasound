import json
import os
import pickle
from functools import partial
from inspect import signature
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy import stats
from scipy.stats import describe
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
from loading.datasets import problem_kind, problem_legal_values, SingleImageDataset, make_att_specs
from loading.img_utils import load_image
from loading.loaders import get_data_for_spec, make_basic_transform
import swifter
from sklearn.metrics import mean_absolute_error, classification_report, roc_auc_score, roc_curve, auc, \
    RocCurveDisplay, precision_score, recall_score

from utils.experiment_utils import get_default_set_spec_dict
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


def run_dummy_baseline(set_name, attribute):
    set_spec_dict = get_default_set_spec_dict()
    set_spec = set_spec_dict[set_name]
    class_values = problem_legal_values[attribute] if attribute in problem_legal_values else None
    patients = get_data_for_spec(set_spec, loader_type='bag', attribute_to_filter=attribute,
                                 legal_attribute_values=class_values,
                                 muscles_to_use=None)

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
    if kind != 'regression':
        mapping = {'NMD': 1, 'no NMD': 0}
        y_true_rv = [mapping[y] for y in y_true]
        train_preds_rv = [mapping[y] for y in train_preds]
        print(roc_auc_score(y_true_rv, train_preds_rv))

    print(scorer(y_true, train_preds))

def compute_EIZ_from_scratch(record, z_score_encoder, root_path, strip_folders=False,
                             mask_root_path=None, factor=None):
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

    load_func = partial(load_image, root_dir=root_path, use_one_channel=True,
                        use_mask=True)

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
    return np.array(mapped_EI)

def get_feature_rep_for_rule_based(EIZ, record):
    # the default representation needs the age of the patient
    additional_features = ['Age']
    feature_rep = compute_exceed_scores(EIZ)
    age_feature = extract_features_from_meta_info(record, additional_features)
    feature_rep = {**feature_rep, **age_feature}
    return feature_rep

def extract_features_from_meta_info(record,features):
    feature_rep = {}
    for additional_feature in features:
        feature_rep[additional_feature] = record.meta_info[additional_feature]
    return feature_rep

def obtain_scores_and_preds(EIZ, record, name):

    feature_rep = get_feature_rep_for_rule_based(EIZ, record)

    new_pred = predict_rule_based(feature_rep)
    feature_rep['pred'] = new_pred
    feature_rep.pop('Age')
    feature_rep['method'] = name
    feature_rep['pid'] = record.meta_info['pid']
    return feature_rep

def process_feature_rep(feature_rep):
    new_rep = {}
    # remove the number, as it could leak info about the patient type
    # via the fact that different diagnostic protocols have different
    # numbers of muscles
    n_scores = feature_rep.pop('NumberZscores')
    for name, value in feature_rep.items():
        if name.startswith('Z_exceed'):
            new_rep[name] = value / n_scores
        else:
            new_rep[name] = value
    return new_rep

def extract_shapiro(vector):
    vector = vector[~np.isnan(vector)]
    if len(vector) < 3:
        return np.nan
    else:
        return stats.shapiro(vector).statistic

def extract_anderson(vector):
    vector = vector[~np.isnan(vector)]
    if len(vector) == 0:
        return np.nan
    return stats.anderson(vector).statistic

def extract_entropy(vector):
    x = vector[~np.isnan(vector)]
    hist_counts, hist_bins = np.histogram(x,bins=5)
    return stats.entropy(hist_counts)

def smooth_vector(vector,range,nbins):
    vector = vector[~np.isnan(vector)]
    bins = np.linspace(range[0],range[1], nbins)
    digitized = np.digitize(vector, bins)
    result = np.array([bins[elem] for elem in digitized])
    return result

extractors = [{'name': 'MEAN','func': np.nanmean, 'scale_inv': False, 'type': 'location'},
              {'name': 'MED','func': np.nanmedian, 'scale_inv': False, 'type': 'location'},
              {'name': 'MIN','func': np.nanmin, 'scale_inv': False, 'type': 'minmax'},
              {'name': 'MAX','func': np.nanmax, 'scale_inv': False, 'type': 'minmax'},
              {'name': 'SD','func': np.nanstd, 'scale_inv': False, 'type': 'dispersion'},
              {'name': 'IQR','func': partial(stats.iqr, nan_policy='omit'), 'scale_inv': False, 'type': 'dispersion'},
              {'name': 'MAD','func': partial(stats.median_abs_deviation, nan_policy='omit'), 'scale_inv': False, 'type': 'dispersion'},

              {'name': 'KUR','func': partial(stats.kurtosis, nan_policy='omit'), 'scale_inv': True, 'type': 'shape'},
              {'name': 'SKEW', 'func': partial(stats.skew, nan_policy='omit'),'scale_inv': True, 'type': 'shape'},
              {'name': 'ENT', 'func': extract_entropy, 'scale_inv': True, 'type': 'shape'},
              {'name': 'SHAP', 'func': extract_shapiro,'scale_inv': True, type: 'goodness'},
              {'name': 'AND', 'func': extract_anderson,'scale_inv': True, type: 'goodness'}]

extractor_frame = pd.DataFrame(extractors)

# options: using EI or EIz as substrate
# groups of features for ablation
def obtain_feature_rep_ml_experiment(set_name, use_eiz=True, ei_extraction_method=None, local=False,
                                     additional_features=None):

    # use the original scores as default
    if not ei_extraction_method:
        ei_extraction_method = partial(get_original_scores)

    set_spec_dict = get_default_set_spec_dict(local=local)
    set_spec = set_spec_dict[set_name]
    patients = get_data_for_spec(set_spec, loader_type='bag', attribute_to_filter='Class',
                                 legal_attribute_values=problem_legal_values['Class'],
                                 muscles_to_use=None)
    feature_reps = []
    if not additional_features:
        additional_features = []

    for patient in patients:
        patient.try_closest_fallback_to_latest()
        record = patient.get_selected_record()
        # allow swapping between z_scores and EI values
        return_dict = ei_extraction_method(record)
        if use_eiz:
            if 'EIZ' not in return_dict:
                raise ValueError(f'Required z-score computation, but method {ei_extraction_method} did not provide it.')
            vector = return_dict['EIZ']
            prefix = 'EIZ'
        else:
            vector = return_dict['EI']
            prefix = 'EI'
        # drop all na vectors
        vector = vector[~np.isnan(vector)]
        if len(vector) == 0:
            continue
        # optionally smooth using different bins
        smoothed_vectors = {}
        smoothing_factors = []
        for smoothing_factor in smoothing_factors:
            smoothed_vectors['smoothed_' + str(smoothing_factor)] = smooth_vector(vector,(0,256),smoothing_factor)
        # always use the original scale
        smoothed_vectors['base'] = vector

        feature_rep = {}
        # additionally filter
        filtered_frame = extractor_frame#[extractor_frame['scale_inv']]
        # feature extraction starts here
        for smoothing_name, smoothed_vector in smoothed_vectors.items():
            for i, row in filtered_frame.iterrows():
                func = row['func']
                if smoothing_name == 'base':
                    name = prefix + '_' + row['name']
                else:
                    name = prefix + '_' + row['name'] + '_' + smoothing_name
                value = func(smoothed_vector)
                feature_rep[name] = value

        demographic_features = extract_features_from_meta_info(record, additional_features)

        feature_rep = {**feature_rep, **demographic_features}

        feature_rep['Class'] = record.meta_info['Class']
        # color = 'r' if feature_rep['Class'] == 'NMD' else 'b'
      #  plt.hist(ei, 5, (0, 150), color=color)
        plt.show()
        feature_reps.append(feature_rep)
    feature_frame = pd.DataFrame(feature_reps)
    return feature_frame

def train_trad_ml_baseline(train_set_name, val_set_name, use_eiz=True, demographic_features=False, local=True):
    additional_features = ['Age', 'Sex', 'BMI'] if demographic_features else []
    train_set = obtain_feature_rep_ml_experiment(train_set_name,use_eiz=use_eiz,local=local, additional_features=additional_features)
    val_set = obtain_feature_rep_ml_experiment(val_set_name,use_eiz=use_eiz, local=local, additional_features=additional_features)
    train_set['Class'] = train_set['Class'].replace({'no NMD': 0, 'NMD': 1})
    val_set['Class'] = val_set['Class'].replace({'no NMD': 0, 'NMD': 1})
    from pycaret.classification import setup, compare_models, set_config, predict_model, save_model, \
        interpret_model, models, tune_model
    models_to_use = models(type='ensemble')
    models_to_use = models_to_use.index.to_list()
    features = set(train_set.columns)
    features.remove('Class')
    # set the experiment up
    exp = setup(train_set, target='Class', numeric_features=features, html = False, session_id = 123, train_size=0.7)
    # sidestep the fact that the the lib makes another validation set
    # manually get the pipeline
    pipeline = exp[7]
    X_train = train_set.drop(columns='Class')
    # transform and set as the training set
    X_train = pipeline.transform(X_train)
    set_config('X_train', X_train)
    set_config('y_train', train_set['Class'])
    X_test = val_set.drop(columns='Class')
    # transform and set as the validation set
    X_test = pipeline.transform(X_test)
    set_config('X_test', X_test)
    set_config('y_test', val_set['Class'])

    best_model = compare_models(whitelist=models_to_use, sort = 'AUC',n_select=1)
    interpret_model(best_model)
    # now, do some additional tuning
    best_model = tune_model(best_model, optimize = 'AUC')
    interpret_model(best_model)

    model_path = train_set_name + '_eiz_' + str(use_eiz) + '_dem_' + str(demographic_features) + '_caret_model'
    save_model(best_model, model_path)
    # get results on val set as dataframe
    results = predict_model(best_model,verbose=False)
  #  print(classification_report(results['Class'], results['Label']))
    best_threshold = evaluate_roc(results['Class'], results['Score'], method='val_set_training')
  #  print(best_model)
    return {'best_threshold' : best_threshold, 'model_path': model_path}

def evaluate_ml_baseline(pkl_path, val_set_name, use_eiz=True, demographic_features=False, local=True,
                         ei_extraction_method=None):
    additional_features = ['Age', 'Sex', 'BMI'] if demographic_features else []
    from pycaret.classification import load_model
    pipeline, model = load_model(pkl_path)

    if not ei_extraction_method:
        ei_extraction_method = get_original_scores

    val_set = obtain_feature_rep_ml_experiment(val_set_name, use_eiz=use_eiz,
                                               additional_features=additional_features,
                                               ei_extraction_method=ei_extraction_method,local=local)
    val_set['Class'] = val_set['Class'].replace({'no NMD': 0, 'NMD': 1})
    X_test = val_set.drop(columns='Class')
    transformed = pipeline.transform(X_test)
    proba = model.predict_proba(transformed)[:, 1]
    return proba

def get_original_scores(record):
    return {'EI': np.array(record.meta_info['EIs']), 'EIZ': np.array(record.meta_info['EIZ'])}

def get_recomputed_z_scores(record, z_score_encoder):
    return_dict = {}
    EIs = np.array(record.meta_info['EIs'])
    return_dict['EI'] = EIs
    if z_score_encoder:
        EIZ = obtain_zscores(EIs, record, z_score_encoder)
        return_dict['EIZ'] = EIZ
    return return_dict

def get_regression_adjusted_z_scores(record,z_score_encoder, regression_model):
    return_dict = {}
    EI_regression = adjust_EI_with_regression(record, regression_model)
    return_dict['EI'] = EI_regression
    if z_score_encoder:
        EIZ_regression = obtain_zscores(EI_regression, record, z_score_encoder)
        return_dict['EIZ'] = EIZ_regression
    return return_dict

def get_brightness_adjusted_z_scores(record, z_score_encoder, factor):
    return_dict = {}
    EI_adjustment = adjust_EI_with_factor(record, factor)
    return_dict['EI'] = EI_adjustment
    if z_score_encoder:
        EIZ_regression = obtain_zscores(EI_adjustment, record, z_score_encoder)
        return_dict['EIZ'] = EIZ_regression
    return return_dict

def extract_y_true(set_name):
    set_spec_dict = get_default_set_spec_dict()
    set_spec = set_spec_dict[set_name]
    patients = get_data_for_spec(set_spec, loader_type='bag', attribute_to_filter='Class',
                                 legal_attribute_values=problem_legal_values['Class'],
                                 muscles_to_use=None)
    y_true = []
    meta_infos = []

    for patient in patients:
        y_true.append(patient.attributes['Class'])
        meta_infos.append(patient.attributes)

    mapping = {'NMD': 1, 'no NMD': 0}
    y_true_rv = [mapping[y] for y in y_true]

    return y_true_rv, pd.DataFrame(meta_infos)

def export_selected_records(set_name):
    set_spec_dict = get_default_set_spec_dict()
    set_spec = set_spec_dict[set_name]

    patients = get_data_for_spec(set_spec, loader_type='bag', attribute_to_filter='Class',
                                 legal_attribute_values=problem_legal_values['Class'],
                                 muscles_to_use=None)
    info_dicts = []
    for patient in patients:
        patient.select_closest()
        record = patient.get_selected_record()
        info_dicts.append(record.meta_info)
    return pd.DataFrame(info_dicts)


def run_rule_based_baseline(set_name, ei_extraction_method):
    set_spec_dict = get_default_set_spec_dict()
    set_spec = set_spec_dict[set_name]
    patients = get_data_for_spec(set_spec, loader_type='bag', attribute_to_filter='Class',
                                 legal_attribute_values=problem_legal_values['Class'],
                                 muscles_to_use=None)


    preds = []
    for patient in tqdm.tqdm(patients):
        patient.try_closest_fallback_to_latest()
        record = patient.get_selected_record()

        eiz = ei_extraction_method(record)['EIZ']

        feature_rep = get_feature_rep_for_rule_based(eiz, record)
        pred = predict_rule_based(feature_rep)
        preds.append(pred)

    preds = pd.Series(preds)

    y_proba_rv = preds.replace({'NMD': 1, 'no NMD': 0, 'unknown or uncertain': 0.5}).values

    return y_proba_rv

def find_threshold_for_sensitivity(y_true,y_pred,sens):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    valid = tpr >= sens
    ind = np.argmax(valid)
    specs = 1 - fpr
    print(f'Specificity obtained: {specs[ind]}')
    return thresholds[ind]

def find_threshold_for_specificity(y_true,y_pred,spec):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    max_fpr = 1- spec
    # find first element that exceeds this fpr
    too_big = fpr > max_fpr
    invalid = np.argmax(too_big)
    ind = invalid - 1
    print(f'Sensitvity obtained: {tpr[ind]}')
    return thresholds[ind]

def evaluate_roc(y_true, y_pred, method, plot=True):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    roc_auc = auc(fpr, tpr)
    # best point on the ROC curve --> Youden's J
    J = tpr - fpr
    best_ind = np.argmax(J)
    best_threshold = thresholds[best_ind]

    print(f'Best threshold: < {np.round(best_threshold,3)} --> negative')

    # compute precision and recall at that threshold
    binarized = (y_pred >= best_threshold).astype(int)
    recall = recall_score(y_true, binarized)
    precision = precision_score(y_true, binarized)

    print(f'Recall = {np.round(recall,3)}, Precision = {np.round(precision,3)}')
    if plot:
        viz = RocCurveDisplay(
            fpr=fpr,
            tpr=tpr,
            roc_auc=roc_auc,
            estimator_name=method
        )

        viz.plot()
        plt.show()

    print(f'AUC: {np.round(roc_auc,3)}')

    return best_threshold

def evaluate_threshold(y_true, y_proba, threshold):
    # compute precision and recall at that threshold
    binarized = (y_proba >= threshold).astype(int)
    return classification_report(y_true, binarized)


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
    total.reset_index(inplace=True, drop=True)

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


def compute_brightness():

    att_spec_dict = make_att_specs()
    set_spec_dict = get_default_set_spec_dict()
    esoate_spec = set_spec_dict['GE_Logiq_E_im_muscle_chart']
    esoate_images = get_data_for_spec(esoate_spec, loader_type='image', dropna_values=False)
   # esoate_images = esoate_images[0:2000]

    transform = make_basic_transform('GE_Logiq_E', limit_image_size=False, to_tensor=True)

    ds = SingleImageDataset(image_frame=esoate_images, root_dir=esoate_spec.img_root_path, attribute_specs=[att_spec_dict['Sex']],
                            return_attribute_dict=False, transform=transform,
                            use_one_channel=True)


    mean, std  = compute_normalization_parameters(ds, 1)

    print(mean)
    print(std)

brightness_dict = {'Philips_iU22': 0.1693, 'ESAOTE_6100': 0.2328, 'GE_Logiq_E': 0.2314}
# source, target, mapping from source to target
lr_dict = {'Philips_iU22': {'ESAOTE_6100': '/home/klux/PycharmProjects/muscle-ultrasound/linear_regression_esaote_to_philips.pkl'},
           'ESAOTE_6100': {'Philips_iU22': '/home/klux/PycharmProjects/muscle-ultrasound/linear_regression_philips_to_esoate.pkl'}}

# A is Esaote, so fake A is the mapped Philips and fake B is mapped Esaote
# for these experiments, we always want to use the mapped target

#mapped_path_dict = {'Philips_iU22': {'ESAOTE_6100': '/mnt/chansey/klaus/pytorch-CycleGAN-and-pix2pix/results/1000samples/fakeA'},
#                    'ESAOTE_6100': {'Philips_iU22': '/mnt/chansey/klaus/pytorch-CycleGAN-and-pix2pix/results/1000samples/fakeB'}}

standard_cyclegan = {'Philips_iU22': {'ESAOTE_6100': '/media/klux/Elements/standard_cyclegan/fakeA'},
                    'ESAOTE_6100': {'Philips_iU22': '/media/klux/Elements/standard_cyclegan/fakeB'}}

cycada_sem_paths = {'Philips_iU22': {'ESAOTE_6100': '/home/klux/Thesis_2/images_cycada/mappedA'},
                    'ESAOTE_6100': {'Philips_iU22': '/home/klux/Thesis_2/images_cycada/mappedB'}}

cycada_no_sem_paths = {'Philips_iU22': {'ESAOTE_6100': '/home/klux/Thesis_2/images_cycada_no_sem/mappedA'},
                    'ESAOTE_6100': {'Philips_iU22': '/home/klux/Thesis_2/images_cycada_no_sem/mappedB'}}

image_mappings = {'standard_cyclegan': standard_cyclegan,
                  'cycada_semantic': cycada_sem_paths,
                  'cycada_no_semantic': cycada_no_sem_paths}

def get_z_score_encoder(source_set):
    if source_set == 'ESAOTE_6100':
        return MuscleZScoreEncoder('data/MUS_EI_models.json')

def get_brightness_factor(source_set,target_set):
    source_bright = brightness_dict[source_set]
    target_bright = brightness_dict[target_set]
    factor = source_bright / target_bright
    return factor

def get_lr_model(source_set, target_set):
    model_path = lr_dict[source_set][target_set]
    with open(model_path, 'rb') as f:
        lr_model = pickle.load(f)
    return lr_model

def get_mapped_path(source_set, target_set, method='standard_cyclegan'):
    mapped_path_dict = image_mappings[method]
    return mapped_path_dict[source_set][target_set]

def get_adjustment_method(type, source_set, target_set, z_score_encoder=None):
    ei_extraction_method = None

    if type == 'brightness':
        factor = get_brightness_factor(source_set.device,target_set.device)
        ei_extraction_method = partial(get_brightness_adjusted_z_scores, z_score_encoder=z_score_encoder, factor=factor)
    if type == 'regression':
        lr_model = get_lr_model(source_set.device,target_set.device)
        ei_extraction_method = partial(get_regression_adjusted_z_scores, z_score_encoder=z_score_encoder,
                                       regression_model=lr_model)
    if type == 'recompute':
        ei_extraction_method = partial(get_recomputed_z_scores, z_score_encoder=z_score_encoder)

    if type == 'mapped_images':
        mapped_path = get_mapped_path(source_set.device, target_set.device)
        mask_root_path = target_set.img_root_path
        ei_extraction_method = partial(compute_EIZ_from_scratch,z_score_encoder=z_score_encoder, root_path=mapped_path,
                                       strip_folders=True, mask_root_path=mask_root_path)

    if type == 'original':
        ei_extraction_method = get_original_scores

    if not ei_extraction_method:
        raise ValueError(f'Unknown method for adjustment: {type}')

    return ei_extraction_method

if __name__ == '__main__':
 #   compute_brightness_diff()
 #   analyze_multi_device_patients()

    set_spec_dict = get_default_set_spec_dict(local=True)

    source = 'ESAOTE_6100'
    target = 'Philips_iU22' # 'GE_Logiq_E_im_muscle_chart'
    evaluate_on = 'test' # can also specify 'test'


    eval_set_name = target + '_' + evaluate_on


    target_set = set_spec_dict[eval_set_name]
    source_set = set_spec_dict[source + '_' + 'train']

    export_path = os.path.join('roc_analysis', eval_set_name)
    os.makedirs(export_path, exist_ok=True)
    # extract ground truth
    y_true, patients = extract_y_true(eval_set_name)
    # export
    pd.DataFrame(y_true, columns=['true']).to_csv(os.path.join(export_path, 'y_true.csv'), index=False, header=True)
    patients.to_pickle(os.path.join(export_path, 'patients.pkl'))
    records = export_selected_records(eval_set_name)
    records.to_pickle(os.path.join(export_path, 'records.pkl'))
    # path for predictions
    proba_path = os.path.join(export_path, 'proba')
    indomain_path = os.path.join(proba_path, 'in_domain')
    outdomain_path = os.path.join(proba_path, 'out_domain')
    os.makedirs(indomain_path, exist_ok=True)
    os.makedirs(outdomain_path, exist_ok=True)
    z_score_encoder = get_z_score_encoder(source)

    evaluate_rules = False

    if evaluate_rules:
        if z_score_encoder:
            ei_extraction_methods = ['original', 'recompute', 'regression', 'brightness']#, 'mapped_images']
        else:
            ei_extraction_methods = ['original']

        # rule-based baseline
        for method_name in ei_extraction_methods:
            ei_extraction_method = get_adjustment_method(method_name, source_set, target_set, z_score_encoder)
            y_proba = run_rule_based_baseline(eval_set_name, ei_extraction_method)
            exp_name = 'rulebased' + '_' + method_name
            base_path = indomain_path if method_name == 'original' else outdomain_path
            csv_path = os.path.join(base_path, (exp_name + '.csv'))
            pd.DataFrame(y_proba, columns=['pred']).to_csv(csv_path, index=False, header=True)
            evaluate_roc(y_true, y_proba, exp_name)

    evaluate_ml = True

    if evaluate_ml:
        # ML based methods

        eiz_options = [False]#[False, True]
        demographic_features = False
        local = True

        for use_eiz in eiz_options:
            # train on source and eval on original val set
            result_dict = train_trad_ml_baseline(source + '_train', source + '_' + 'val', use_eiz=use_eiz,
                                                 local=local, demographic_features=demographic_features)

            source_eval_set = source + '_' + evaluate_on
            # evaluate on source
            y_proba = evaluate_ml_baseline(result_dict['model_path'], source_eval_set,
                                 use_eiz=use_eiz, local=local, demographic_features=demographic_features)

            exp_name = result_dict['model_path'] + '_' + 'original'
            y_true_source, patients_source = extract_y_true(source_eval_set)
            evaluate_roc(y_true_source, y_proba, exp_name)

            export_path = os.path.join('roc_analysis', source_eval_set, 'proba', 'in_domain')
            os.makedirs(export_path, exist_ok=True)
            csv_path = os.path.join(export_path, (exp_name + '.csv'))
            pd.DataFrame(y_proba, columns=['pred']).to_csv(csv_path, index=False, header=True)

            # evaluate on target
            if use_eiz:
                z_score_encoder = get_z_score_encoder(source)
            else:
                z_score_encoder = None

            if use_eiz and not z_score_encoder:
                ei_extraction_methods = ['original']
            else:
                ei_extraction_methods = ['original', 'recompute', 'regression', 'brightness']#, 'mapped_images']

            for method_name in ei_extraction_methods:
                ei_extraction_method = get_adjustment_method(method_name, source_set, target_set, z_score_encoder)

                y_proba = evaluate_ml_baseline(result_dict['model_path'], eval_set_name, use_eiz=use_eiz, local=local, demographic_features=demographic_features,
                                               ei_extraction_method=ei_extraction_method)
                exp_name = result_dict['model_path'] + '_' + method_name
                # export
                csv_path = os.path.join(proba_path, 'out_domain', (exp_name + '.csv'))
                pd.DataFrame(y_proba, columns=['pred']).to_csv(csv_path, index=False, header=True)

                # performance
                evaluate_roc(y_true, y_proba, exp_name)

                # performance at best threshold during training
                previous_threshold = result_dict['best_threshold']
                if previous_threshold:
                    print('Performance at best previous threshold')
                    print(evaluate_threshold(y_true, y_proba, previous_threshold))

            # can only do GE evaluation if no z-scores required
            if not use_eiz:
               factor = get_brightness_factor(source_set.device,'GE_Logiq_E')
               # separate evaluation logic for GE is necessary
               ei_extraction_method = partial(compute_EIZ_from_scratch, z_score_encoder=z_score_encoder,
                                              root_path='/home/klux/Thesis_2/klaus/myositis/processed_imgs',
                                              factor=factor)

               y_proba = evaluate_ml_baseline(result_dict['model_path'], 'GE_Logiq_E_im_muscle_chart', use_eiz=False, local=local,
                                    demographic_features=demographic_features,
                                    ei_extraction_method=ei_extraction_method)

               exp_name = result_dict['model_path']
               # export
               export_path_ge = os.path.join('roc_analysis', 'GE_Logiq_E_im_muscle_chart', 'proba')
               os.makedirs(export_path_ge, exist_ok=True)
               csv_path = os.path.join(export_path_ge, 'out_domain', (exp_name + '_brightness' + '.csv'))
               # performance

               pd.DataFrame(y_proba, columns=['pred']).to_csv(csv_path, index=False, header=True)
