import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .rk import RK


class PLS(PLSRegression):

    def transform(self, X, Y=None, copy=True):
        return PLSRegression.predict(self, X, copy)


def fdatagrid_with_resolution(array, resolution):
    step = (array.data_matrix.shape[1] - 1) // 2**resolution

    return array[:, ::step]


def dict_with_resolution(d, resolution):
    d_new = {}

    for key, value in d.items():
        d_new[key] = fdatagrid_with_resolution(value, resolution)

    return d_new


def classifier_lda(n_features, cv):
    return LinearDiscriminantAnalysis(priors=[.5, .5])


def classifier_qda(n_features, cv):
    return QuadraticDiscriminantAnalysis(priors=[.5, .5])


def classifier_pls_centroid(n_features, cv):
    return GridSearchCV(Pipeline([
        ("scaler", StandardScaler()),
        ("pls", PLS()),
        ("centroid", NearestCentroid())]),
        param_grid={
        "pls__n_components": range(1, min(21, n_features))
    }, cv=cv)


def classifier_pca_qda(n_features, cv):
    return GridSearchCV(Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(random_state=0)),
        ("qda", QuadraticDiscriminantAnalysis())]),
        param_grid={
        "pca__n_components": range(1, min(21, n_features))
    }, cv=cv)


def classifier_rkc(n_features, cv):
    return GridSearchCV(Pipeline([
        ("rk", RK()),
        ("lda", LinearDiscriminantAnalysis())]),
        param_grid={
        "rk__n_components": range(1, min(21, n_features))
    }, cv=cv)


def compute_scores_list(clf, X_train_w_res_list, y_train_list,
                        X_test_w_res_list, y_test_list, cv=10):

    scores = []
    confusion_matrices = []

    if X_test_w_res_list is None:
        # Cross validation
        for (X_train_w_res, y_train) in zip(
                X_train_w_res_list, y_train_list):
            predicted = cross_val_predict(
                clf, X_train_w_res, y_train, cv=cv)
            scores.append(accuracy_score(y_train, predicted))
            confusion_matrices.append(confusion_matrix(
                y_train, predicted))
    else:
        # Score in test
        for (X_train_w_res, y_train, X_test_w_res, y_test) in zip(
                X_train_w_res_list, y_train_list,
                X_test_w_res_list, y_test_list):

            clf.fit(X_train_w_res, y_train)
            scores.append(clf.score(X_test_w_res, y_test))
            confusion_matrices.append(confusion_matrix(
                y_test, clf.predict(X_test_w_res)))

    return scores, confusion_matrices


def classification_test_common(X_train_list, y_train_list,
                               X_test_list=None, y_test_list=None,
                               *,
                               max_pow, _run,
                               additional_classifiers_fd={},
                               additional_classifiers_matrix={},
                               start_pow=1,
                               no_common_classifiers=False):

    cv = 10

    scores = {}
    confusion_matrices = {}
    classifiers_fd = {
        **additional_classifiers_fd
    }

    if no_common_classifiers:
        classifiers_matrix = {
            **additional_classifiers_matrix
        }
    else:
        classifiers_matrix = {
            'lda': classifier_lda,
            'qda': classifier_qda,
            'pls_centroid': classifier_pls_centroid,
            'pca_qda': classifier_pca_qda,
            'rkc': classifier_rkc,
            **additional_classifiers_matrix
        }

    classifiers_all = {**classifiers_fd, **classifiers_matrix}

    for key in classifiers_all:
        scores[key] = [np.nan] * (max_pow + 1)
        confusion_matrices[key] = [np.nan] * (max_pow + 1)

    for resolution in range(start_pow, max_pow + 1):

        X_train_w_res_list = [fdatagrid_with_resolution(X_train, resolution)
                              for X_train in X_train_list]
        X_train_w_res_list_matrices = [
            X.data_matrix[..., 0][:, 1:] for X in X_train_w_res_list]

        if X_test_list is not None:
            X_test_w_res_list = [fdatagrid_with_resolution(X_test, resolution)
                                 for X_test in X_test_list]
            X_test_w_res_list_matrices = [
                X.data_matrix[..., 0][:, 1:] for X in X_test_w_res_list]

        n_features = len(X_train_w_res_list[0].sample_points[0])

        for key, value in classifiers_all.items():
            clf = value(n_features=n_features, cv=cv)

            if key in classifiers_fd:
                X_train = X_train_w_res_list
            else:
                X_train = X_train_w_res_list_matrices

            if X_test_list is not None:
                if key in classifiers_fd:
                    X_test = X_test_w_res_list
                else:
                    X_test = X_test_w_res_list_matrices
            else:
                X_test = None

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
        scores[key] = np.array(scores[key][start_pow:], ndmin=2)

    _run.info['scores'] = scores
    _run.info['confusion_matrices'] = confusion_matrices

    return scores, confusion_matrices
