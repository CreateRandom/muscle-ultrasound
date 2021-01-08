import pandas as pd
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import mean_absolute_error, classification_report, roc_auc_score

from loading.datasets import problem_legal_values, problem_kind
from loading.loaders import get_data_for_spec
from utils.experiment_utils import get_default_set_spec_dict


def extract_features_from_meta_info(record, features):
    """A small helper to extract additional features from a patient record."""
    feature_rep = {}
    for additional_feature in features:
        feature_rep[additional_feature] = record.meta_info[additional_feature]
    return feature_rep


def export_selected_records(set_name):
    '''For the patients in this set, get the relevant record for each and then store them all in one DataFrame.'''
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


def extract_y_true(set_name):
    '''Extract ground truth NMD diagnosis values for each patient in this set.'''
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


def run_dummy_baseline(set_name, attribute):
    '''
    Runs a simple baseline that always predicts the most frequent value for a given attribute.
    :param set_name: the name of the dataset to run on
    :param attribute: the attribute to predict
    :return: a score reflecting the performance
    '''
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

    return scorer(y_true, train_preds)