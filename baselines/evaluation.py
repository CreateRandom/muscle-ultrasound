import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, RocCurveDisplay, classification_report


def find_threshold_for_sensitivity(y_true,y_pred,sens):
    '''For a desired sensitivity level, find the necessary threshold.'''
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    valid = tpr >= sens
    ind = np.argmax(valid)
    specs = 1 - fpr
    print(f'Specificity obtained: {specs[ind]}')
    return thresholds[ind]


def find_threshold_for_specificity(y_true,y_pred,spec):
    '''For a desired specificity level, find the necessary threshold.'''
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    max_fpr = 1- spec
    # find first element that exceeds this fpr
    too_big = fpr > max_fpr
    invalid = np.argmax(too_big)
    ind = invalid - 1
    print(f'Sensitvity obtained: {tpr[ind]}')
    return thresholds[ind]


def evaluate_roc(y_true, y_pred, method, plot=True):
    '''A quick helper for ad-hoc ROC analysis, more seasoned comparison later in R.'''
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
    '''Compute precision and recall at that threshold. '''
    binarized = (y_proba >= threshold).astype(int)
    return classification_report(y_true, binarized)