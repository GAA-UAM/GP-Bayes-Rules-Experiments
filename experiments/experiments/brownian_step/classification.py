import tempfile

from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)

import matplotlib.pyplot as plt
import numpy as np

from . import experiment
from ..common.classification import (fdatagrid_with_resolution,
                                     classifier_galeano,
                                     classifier_rkc,
                                     classifier_pls_centroid, plot_with_var)
from .brownian_step_classifier import BrownianStepClassifier


def compute_scores_list(clf, X_train_w_res_list, y_train_list,
                        X_test_w_res_list, y_test_list):
    return [clf.fit(
        X_train_w_res, y_train).score(
            X_test_w_res, y_test)
        for (X_train_w_res, y_train,
             X_test_w_res, y_test)
        in zip(X_train_w_res_list, y_train_list,
               X_test_w_res_list, y_test_list)]


@experiment.capture
def classification_test(X_train_list, y_train_list, X_test_list, y_test_list,
                        max_pow, _run):

    # Leave one-out
    cv = 10

    scores = [np.nan] * (max_pow + 1)
    scores_lda = [np.nan] * (max_pow + 1)
    scores_qda = [np.nan] * (max_pow + 1)
    scores_pls_centroid = [np.nan] * (max_pow + 1)
    scores_galeano = [np.nan] * (max_pow + 1)
    scores_rkc = [np.nan] * (max_pow + 1)

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

        clf = BrownianStepClassifier()
        clf_lda = LinearDiscriminantAnalysis(
            priors=[.5, .5])
        clf_qda = QuadraticDiscriminantAnalysis(
            priors=[.5, .5])
        clf_pls_centroid = classifier_pls_centroid(
            n_features=n_features,
            cv=cv)
        clf_galeano = classifier_galeano(
            n_features=n_features,
            cv=cv)
        clf_rkc = classifier_rkc(
            n_features=n_features,
            cv=cv)

        scores[resolution] = compute_scores_list(
            clf, X_train_w_res_list, y_train_list,
            X_test_w_res_list, y_test_list)
        scores_lda[resolution] = compute_scores_list(
            clf_lda, X_train_w_res_list_matrices, y_train_list,
            X_test_w_res_list_matrices, y_test_list)
        scores_qda[resolution] = compute_scores_list(
            clf_lda, X_train_w_res_list_matrices, y_train_list,
            X_test_w_res_list_matrices, y_test_list)
        scores_pls_centroid[resolution] = compute_scores_list(
            clf_pls_centroid, X_train_w_res_list_matrices, y_train_list,
            X_test_w_res_list_matrices, y_test_list)
        scores_galeano[resolution] = compute_scores_list(
            clf_galeano, X_train_w_res_list_matrices, y_train_list,
            X_test_w_res_list_matrices, y_test_list)
        scores_rkc[resolution] = compute_scores_list(
            clf_rkc, X_train_w_res_list_matrices, y_train_list,
            X_test_w_res_list_matrices, y_test_list)

        _run.log_scalar("scores", np.mean(scores[resolution]), resolution)
        _run.log_scalar("scores_lda", np.mean(
            scores_lda[resolution]), resolution)
        _run.log_scalar("scores_qda", np.mean(
            scores_qda[resolution]), resolution)
        _run.log_scalar("scores_pls_centroid",
                        np.mean(scores_pls_centroid[resolution]), resolution)
        _run.log_scalar("scores_galeano",
                        np.mean(scores_galeano[resolution]), resolution)
        _run.log_scalar("scores_rkc", np.mean(
            scores_rkc[resolution]), resolution)

    scores = np.array(scores[1:], ndmin=2)
    scores_lda = np.array(scores_lda[1:], ndmin=2)
    scores_qda = np.array(scores_qda[1:], ndmin=2)
    scores_pls_centroid = np.array(scores_pls_centroid[1:], ndmin=2)
    scores_galeano = np.array(scores_galeano[1:], ndmin=2)
    scores_rkc = np.array(scores_rkc[1:], ndmin=2)

    _run.info['scores'] = scores
    _run.info['scores_lda'] = scores_lda
    _run.info['scores_qda'] = scores_qda
    _run.info['scores_pls_centroid'] = scores_pls_centroid
    _run.info['scores_galeano'] = scores_galeano
    _run.info['scores_rkc'] = scores_rkc

    mean_scores = np.mean(scores, axis=1)
    mean_scores_lda = np.mean(scores_lda, axis=1)
    mean_scores_qda = np.mean(scores_qda, axis=1)
    mean_scores_pls_centroid = np.mean(scores_pls_centroid, axis=1)
    mean_scores_galeano = np.mean(scores_galeano, axis=1)
    mean_scores_rkc = np.mean(scores_rkc, axis=1)

    std_scores = np.std(scores, axis=1)
    std_scores_lda = np.std(scores_lda, axis=1)
    std_scores_qda = np.std(scores_qda, axis=1)
    std_scores_pls_centroid = np.std(scores_pls_centroid, axis=1)
    std_scores_galeano = np.std(scores_galeano, axis=1)
    std_scores_rkc = np.std(scores_rkc, axis=1)

    legend_scores = 'Step-Rule'
    legend_scores_lda = 'LDA'
    legend_scores_qda = 'QDA'
    legend_scores_pls_centroid = 'PLS+Centroid'
    legend_scores_galeano = 'PCA+QDA'
    legend_scores_rkc = 'RKC'

    # plt.title('Accuracy')
    plt.figure()

    std_span = 1

    plot_with_var(mean=mean_scores, std=std_scores,
                  std_span=std_span,
                  label=legend_scores, color='C0', linestyle=':', marker='o')
    plot_with_var(mean=mean_scores_qda, std=std_scores_qda,
                  label=legend_scores_qda,
                  color='C2', linestyle='-.', marker='v')
    plot_with_var(mean=mean_scores_lda, std=std_scores_lda,
                  std_span=std_span,
                  label=legend_scores_lda, color='C3', marker='s')
    plot_with_var(mean=mean_scores_pls_centroid, std=std_scores_pls_centroid,
                  std_span=std_span,
                  label=legend_scores_pls_centroid, color='C5', marker='X')
    plot_with_var(mean=mean_scores_galeano, std=std_scores_galeano,
                  std_span=std_span,
                  label=legend_scores_galeano, color='C6', marker='p')
    plot_with_var(mean=mean_scores_rkc, std=std_scores_rkc,
                  std_span=std_span,
                  label=legend_scores_rkc, color='C7', marker='*')
    plt.xticks(*list(zip(*[(i - 1, 2**i)
                           for i in range(1, max_pow + 1)])))
    plt.xlabel("$N_b$")
    plt.ylabel("Accuracy")
    leg = plt.legend(loc="upper left")
    leg.get_frame().set_alpha(1)

    plt.xlim(0, max_pow)
    plt.ylim(top=1.05)
    plt.axhline(1, linestyle=':', color='black')

    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmpfile:
        plt.savefig(tmpfile, format="pdf")
        _run.add_artifact(tmpfile.name, name="plot.pdf")
