from functools import partial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pycaret.classification import models, setup, set_config, compare_models, interpret_model, tune_model, save_model, \
    predict_model
from scipy import stats

from baselines.evaluation import evaluate_roc
from baselines.utils import extract_features_from_meta_info
from baselines.zscores import get_original_scores
from loading.datasets import problem_legal_values
from loading.loaders import get_data_for_spec
from utils.experiment_utils import get_default_set_spec_dict


def extract_shapiro(vector):
    """ Compute the Shapiro statistic of a given vector. """
    vector = vector[~np.isnan(vector)]
    if len(vector) < 3:
        return np.nan
    else:
        return stats.shapiro(vector).statistic


def extract_anderson(vector):
    """ Compute the Anderson statistic of a given vector. """
    vector = vector[~np.isnan(vector)]
    if len(vector) == 0:
        return np.nan
    return stats.anderson(vector).statistic


def extract_entropy(vector):
    """ Compute the entropy of a given vector. """
    x = vector[~np.isnan(vector)]
    hist_counts, hist_bins = np.histogram(x,bins=5)
    return stats.entropy(hist_counts)


def smooth_vector(vector,range,nbins):
    '''
    Reduce the number of different values found in the vector by smoothing it.
    :param vector: The vector to smooth
    :param range: The range to sample replacement bin values from.
    :param nbins: The number of bins, i.e. replacement values to use.
    :return: The smoothed vector.
    '''
    # remove nan elements
    vector = vector[~np.isnan(vector)]
    # create nbins in the provided range
    bins = np.linspace(range[0],range[1], nbins)
    # assign each real-value to the associated bin
    digitized = np.digitize(vector, bins)
    # replace each value by its bin value
    result = np.array([bins[elem] for elem in digitized])
    return result


# The distributional features to extract

# A description of the features we want to extract from each list of EI scores for different muscles
# Each entry contains the name, the function applied to the list and two additional attributes which can be used for
# filtering the features that are extracted.
# Scale-invariant features don't change when the distribution is scaled. The type attribute describes what type of
# parameter the given feature is
feature_extractors = [{'name': 'MEAN', 'func': np.nanmean, 'scale_inv': False, 'type': 'location'},
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

# store it in a dataframe
extractor_frame = pd.DataFrame(feature_extractors)


def obtain_feature_rep_ml_experiment(set_name, use_eiz=True, ei_extraction_method=None, additional_features=None):
    '''
    A method that maps the entire provided set into the feature representation used for the Trad ML experiments.
    :param set_name: The name of the dataset to be mapped.
    :param use_eiz: Use EIZ scores? If false, use raw EI scores.
    :param ei_extraction_method: The method for extracting EI scores from records. Can use original scores or recompute.
    :param additional_features: Additional demographic features to be included (extracted from the records)
    :return: A DataFrame of mapped patient records for classification.
    '''

    # use the original scores as default
    if not ei_extraction_method:
        ei_extraction_method = partial(get_original_scores)

    set_spec_dict = get_default_set_spec_dict()
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


def train_trad_ml_baseline(train_set_name, val_set_name, use_eiz=True, demographic_features=False):
    '''
    Trains a ensemble based classifier on a distribution based feature representation of EI or EIZ scores to predict
    whether or not a patient has an NMD
    :param train_set_name: The name of the training set to use
    :param val_set_name: The name of the validation set to use
    :param use_eiz: Whether to use EIZ or raw EI scores
    :param demographic_features: Whether to include demographic features.
    :return: A dictionary with the path to the stored model and its best operating threshold.
    '''
    additional_features = ['Age', 'Sex', 'BMI'] if demographic_features else []
    # obtain feature representations
    train_set = obtain_feature_rep_ml_experiment(train_set_name, use_eiz=use_eiz,
                                                 additional_features=additional_features)
    val_set = obtain_feature_rep_ml_experiment(val_set_name, use_eiz=use_eiz, additional_features=additional_features)
    # map to real-valued
    train_set['Class'] = train_set['Class'].replace({'no NMD': 0, 'NMD': 1})
    val_set['Class'] = val_set['Class'].replace({'no NMD': 0, 'NMD': 1})
    # use only ensemble models
    models_to_use = models(type='ensemble')
    models_to_use = models_to_use.index.to_list()
    # get the set of all features in the dataset
    features = set(train_set.columns)
    features.remove('Class')

    # set the experiment up
    exp = setup(train_set, target='Class', numeric_features=features, html = False, session_id = 123, train_size=0.7)
    # sidestep the fact that the the lib makes another validation set

    # manually get the pipeline pycaret uses for transforming the data
    pipeline = exp[7]
    X_train = train_set.drop(columns='Class')
    # transform into the format pycaret expects
    X_train = pipeline.transform(X_train)
    # overwrite the selected train set to use the entire training set instead
    set_config('X_train', X_train)
    set_config('y_train', train_set['Class'])
    # same logic with the val set, use our own instead of the pre-sliced one
    X_test = val_set.drop(columns='Class')
    # transform and set as the validation set
    X_test = pipeline.transform(X_test)
    # overwrite config
    set_config('X_test', X_test)
    set_config('y_test', val_set['Class'])

    # obtain the best model from the list, sorted by val set AUC
    best_model = compare_models(whitelist=models_to_use, sort = 'AUC',n_select=1)
    # interpretability output, get SHAP plots to judge feature importance
    interpret_model(best_model)

    # now, do some additional tuning, compare different hyperparemters, maximize AUC
    best_model = tune_model(best_model, optimize = 'AUC')
    # interpret the best model
    interpret_model(best_model)
    # the path to save the model at
    model_path = get_model_name(train_set_name, use_eiz, demographic_features)
    # save the model
    save_model(best_model, model_path)
    # get results on val set as dataframe
    results = predict_model(best_model,verbose=False)
    # get the threshold at which the model performed best on the val set
    best_threshold = evaluate_roc(results['Class'], results['Score'], method='val_set_training')

    return {'best_threshold' : best_threshold, 'model_path': model_path}

def get_model_name(train_set_name, use_eiz, demographic_features):
    return train_set_name + '_eiz_' + str(use_eiz) + '_dem_' + str(demographic_features) + '_caret_model'


def evaluate_ml_baseline(pkl_path, eval_set_name, use_eiz=True, demographic_features=False, ei_extraction_method=None):
    '''
    Evaluate the a stored ML model on a set.
    :param pkl_path: The path of the trained model to evaluate.
    :param eval_set_name: The name of the set to evaluate on.
    :param use_eiz: Whether to use EIZ or raw EI scores
    :param demographic_features: Whether to include demographic features.
    :param ei_extraction_method: The method for extracting EI scores from records. Can use original scores or recompute.
    :return: Probabilities of NMD for each of the val set patients according to the trained model.
    '''
    additional_features = ['Age', 'Sex', 'BMI'] if demographic_features else []
    from pycaret.classification import load_model
    pipeline, model = load_model(pkl_path)

    if not ei_extraction_method:
        ei_extraction_method = get_original_scores

    val_set = obtain_feature_rep_ml_experiment(eval_set_name, use_eiz=use_eiz, ei_extraction_method=ei_extraction_method,
                                               additional_features=additional_features)
    val_set['Class'] = val_set['Class'].replace({'no NMD': 0, 'NMD': 1})
    X_test = val_set.drop(columns='Class')
    transformed = pipeline.transform(X_test)
    proba = model.predict_proba(transformed)[:, 1]
    return proba