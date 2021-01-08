import numpy as np
import pandas as pd
import tqdm

from baselines.utils import extract_features_from_meta_info
from loading.datasets import problem_legal_values
from loading.loaders import get_data_for_spec
from utils.experiment_utils import get_default_set_spec_dict


def compute_exceed_scores(Z_score_vector):
    """Converts a vector of z scores into a vector of counts for exceeded thresholds."""
    to_return = {}
    to_return['NumberZscores'] = len(Z_score_vector)

    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    for t in thresholds:
        count = sum(Z_score_vector > t)
        name = 'Z_exceed' + str(t)
        to_return[name] = count
    return to_return


def get_feature_rep_for_rule_based(EIZ, record):
    """Build the feature representation vector for the rule-based method."""
    # the default representation needs the age of the patient
    additional_features = ['Age']
    feature_rep = compute_exceed_scores(EIZ)
    age_feature = extract_features_from_meta_info(record, additional_features)
    feature_rep = {**feature_rep, **age_feature}
    return feature_rep


def run_rule_based_baseline(set_name, ei_extraction_method):
    """
    Get the rule-based prediction for each patient in this set.
    :param set_name: The data set to use.
    :param ei_extraction_method: The method to use for adjust EIZ scores.
    :return: Rule-based predictions.
    """
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
    # the rule-based model can also predict uncertain disease state, map this to 0.5 to allow an
    # additional threshold during ROC computation.
    y_proba_rv = preds.replace({'NMD': 1, 'no NMD': 0, 'unknown or uncertain': 0.5}).values

    return y_proba_rv


def predict_rule_based(dict_with_exceed):
    '''The Python version of the Z-score to diagnosis flowchart used for diagnosis in the RUMC neurology department.'''
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