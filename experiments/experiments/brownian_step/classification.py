from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)

import matplotlib.pyplot as plt

import numpy as np

from ..common.classification import (fdatagrid_with_resolution,
                                     classifier_galeano,
                                     classifier_rkc, classifier_pca_centroid,
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


def classification_test(X_train_list, y_train_list, X_test_list, y_test_list,
                        max_pow):

    # Leave one-out
    cv = 10

    scores = [None] * (max_pow + 1)
    scores_lda = [None] * (max_pow + 1)
    scores_pca_centroid = [None] * (max_pow + 1)
    scores_pls_centroid = [None] * (max_pow + 1)
    scores_galeano = [None] * (max_pow + 1)
    scores_rkc = [None] * (max_pow + 1)

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
        clf_pca_centroid = classifier_pca_centroid(
            n_features=n_features,
            cv=cv)
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
        scores_pca_centroid[resolution] = compute_scores_list(
            clf_pca_centroid, X_train_w_res_list_matrices, y_train_list,
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

    scores = np.array(scores[1:])
    scores_lda = np.array(scores_lda[1:])
    scores_pca_centroid = np.array(scores_pca_centroid[1:])
    scores_pls_centroid = np.array(scores_pls_centroid[1:])
    scores_galeano = np.array(scores_galeano[1:])
    scores_rkc = np.array(scores_rkc[1:])

    mean_scores = np.mean(scores, axis=1)
    mean_scores_lda = np.mean(scores_lda, axis=1)
    mean_scores_pca_centroid = np.mean(scores_pca_centroid, axis=1)
    mean_scores_pls_centroid = np.mean(scores_pls_centroid, axis=1)
    mean_scores_galeano = np.mean(scores_galeano, axis=1)
    mean_scores_rkc = np.mean(scores_rkc, axis=1)

    std_scores = np.std(scores, axis=1)
    std_scores_lda = np.std(scores_lda, axis=1)
    std_scores_pca_centroid = np.std(scores_pca_centroid, axis=1)
    std_scores_pls_centroid = np.std(scores_pls_centroid, axis=1)
    std_scores_galeano = np.std(scores_galeano, axis=1)
    std_scores_rkc = np.std(scores_rkc, axis=1)

    legend_scores = 'Step-Rule'
    legend_scores_lda = 'LDA'
    legend_scores_pca_centroid = 'PCA-Centroid'
    legend_scores_pls_centroid = 'PLS-Centroid'
    legend_scores_galeano = 'Galeano'
    legend_scores_rkc = 'RKC'

    # plt.title('Accuracy')
    plt.figure()

    std_span = 1

    plot_with_var(mean=mean_scores, std=std_scores,
                  std_span=std_span,
                  label=legend_scores, color='C0', linestyle=':', marker='o')
    plot_with_var(mean=mean_scores_lda, std=std_scores_lda,
                  std_span=std_span,
                  label=legend_scores_lda, color='C3', marker='s')
    plot_with_var(mean=mean_scores_pca_centroid, std=std_scores_pca_centroid,
                  std_span=std_span,
                  label=legend_scores_pca_centroid, color='C4', marker='P')
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
