
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.metrics import confusion_matrix

import numpy as np

from . import experiment
from . import plot
from ..common.classification import (fdatagrid_with_resolution,
                                     classifier_lda,
                                     classifier_qda,
                                     classifier_pca_qda,
                                     classifier_rkc,
                                     classifier_pls_centroid)
from .brownian_step_classifier import BrownianStepClassifier


def compute_scores_list(clf, X_train_w_res_list, y_train_list,
                        X_test_w_res_list, y_test_list):

    scores = []
    confusion_matrices = []

    for (X_train_w_res, y_train, X_test_w_res, y_test) in zip(
            X_train_w_res_list, y_train_list, X_test_w_res_list, y_test_list):
        clf.fit(X_train_w_res, y_train)
        scores.append(clf.score(X_test_w_res, y_test))
        confusion_matrices.append(confusion_matrix(
            y_test, clf.predict(X_test_w_res)))

    return scores, confusion_matrices


@experiment.capture
def classification_test(X_train_list, y_train_list, X_test_list, y_test_list,
                        max_pow, _run):

    # Leave one-out
    cv = 10

    scores = {}
    confusion_matrices = {}
    classifiers_fd = {
        'optimal': lambda **kwargs: BrownianStepClassifier()
    }
    classifiers_matrix = {
        'lda': classifier_lda,
        'qda': classifier_qda,
        'pls_centroid': classifier_pls_centroid,
        'pca_qda': classifier_pca_qda,
        'rkc': classifier_rkc
    }

    classifiers_all = {**classifiers_fd, **classifiers_matrix}

    for key in classifiers_all:
        scores[key] = [np.nan] * (max_pow + 1)
        confusion_matrices[key] = [np.nan] * (max_pow + 1)

    for resolution in range(1, max_pow + 1):
        X_train_w_res_list = [fdatagrid_with_resolution(X_train, resolution)
                              for X_train in X_train_list]
        X_test_w_res_list = [fdatagrid_with_resolution(X_test, resolution)
                             for X_test in X_test_list]
        X_train_w_res_list_matrices = [
            X.data_matrix[..., 0][:, 1:] for X in X_train_w_res_list]
        X_test_w_res_list_matrices = [
            X.data_matrix[..., 0][:, 1:] for X in X_test_w_res_list]

        n_features = len(X_train_w_res_list[0].sample_points[0])

        for key, value in classifiers_all.items():
            clf = value(n_features=n_features, cv=cv)

            if key in classifiers_fd:
                X_train = X_train_w_res_list
                X_test = X_test_w_res_list
            else:
                X_train = X_train_w_res_list_matrices
                X_test = X_test_w_res_list_matrices

            s, cf = compute_scores_list(
                clf, X_train, y_train_list,
                X_test, y_test_list)
            scores[key][resolution] = s
            confusion_matrices[key][resolution] = cf

            _run.log_scalar("scores_" + key,
                            np.mean(scores[key][resolution]), resolution)

            _run.info['scores'] = scores
            _run.info['confusion_matrices'] = confusion_matrices

    for key in classifiers_all:
        scores[key] = np.array(scores[key][1:], ndmin=2)
        print(scores[key].shape)

    _run.info['scores'] = scores
    _run.info['confusion_matrices'] = confusion_matrices

    plot.plot_scores(max_pow=max_pow,
                     scores=scores,
                     legend_scores_optimal='Step-Rule',
                     _run=_run)
