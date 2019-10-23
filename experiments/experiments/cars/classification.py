from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import numpy as np
import skfda

from ..common.classification import (dict_with_resolution, classifier_galeano,
                                     classifier_rkc,
                                     classifier_pls_centroid, plot_with_var)
from ..common.longvar import LongitudinalVarianceClassifier
from .theoretical import TheoreticalBounds


def build_dataset(d):
    X = []
    y = []

    for i, (_, value) in enumerate(d.items()):

        X.append(value)
        y.append(np.full(shape=value.shape[0], fill_value=i))

    data_matrix = np.vstack([x.data_matrix for x in X])
    X = skfda.FDataGrid(data_matrix=data_matrix,
                        sample_points=X[0].sample_points)

    y = np.concatenate(y)

    return X, y


def classification_test(data, n_points_segment_pow,
                        class_variances, savefig=None):

    # Leave one-out
    cv = 10

    scores = [None] * (n_points_segment_pow + 1)
    scores_real_bayes = [None] * (n_points_segment_pow + 1)
    scores_real_bayes_synt = [None] * (n_points_segment_pow + 1)
    scores_lda = [None] * (n_points_segment_pow + 1)
    scores_pls_centroid = [None] * (n_points_segment_pow + 1)
    scores_galeano = [None] * (n_points_segment_pow + 1)
    scores_rkc = [None] * (n_points_segment_pow + 1)
    mean_scores_theoretical = [None] * (n_points_segment_pow + 1)
    std_scores_theoretical = [None] * (n_points_segment_pow + 1)

    for resolution in range(n_points_segment_pow + 1):
        d = dict_with_resolution(data, resolution)
        X, y = build_dataset(d)

        clf = LongitudinalVarianceClassifier()
        clf_real_bayes = QuadraticDiscriminantAnalysis(
            priors=[.5, .5])
        clf_real_bayes_synt = LongitudinalVarianceClassifier(
            real_bayes_rule=True, synthetic_covariance=True)
        clf_lda = LinearDiscriminantAnalysis(priors=[.5, .5])
        clf_pls_centroid = classifier_pls_centroid(
            n_features=len(X.sample_points[0]), cv=cv)
        clf_galeano = classifier_galeano(
            n_features=len(X.sample_points[0]), cv=cv)
        clf_rkc = classifier_rkc(
            n_features=len(X.sample_points[0]), cv=cv)

        scores[resolution] = cross_val_score(clf, X, y, cv=cv)
        scores_real_bayes[resolution] = cross_val_score(
            clf_real_bayes, X, y, cv=cv)
        scores_real_bayes_synt[resolution] = cross_val_score(
            clf_real_bayes_synt, X, y, cv=cv)
        scores_lda[resolution] = cross_val_score(
            clf_lda, X.data_matrix[..., 0][:, 1:], y, cv=cv)
        scores_pls_centroid[resolution] = cross_val_score(
            clf_pls_centroid, X.data_matrix[..., 0][:, 1:], y, cv=cv)
        scores_galeano[resolution] = cross_val_score(
            clf_galeano, X.data_matrix[..., 0][:, 1:], y, cv=cv)
        scores_rkc[resolution] = cross_val_score(
            clf_rkc, X.data_matrix[..., 0][:, 1:], y, cv=cv)

        mean_0 = 0
        mean_1 = 0

        theoretical_bounds = TheoreticalBounds(variance_0=class_variances[0],
                                               variance_1=class_variances[1],
                                               mean_0=mean_0, mean_1=mean_1,
                                               n_points=X.data_matrix.shape[1])
        mean_scores_theoretical[resolution] = float(
            1 - theoretical_bounds.error_probability())
        std_scores_theoretical[resolution] = float(
            theoretical_bounds.error_probability_std(n_traj=X.shape[0]))

    scores = np.array(scores)
    scores_real_bayes = np.array(scores_real_bayes)
    scores_real_bayes_synt = np.array(scores_real_bayes_synt)
    scores_lda = np.array(scores_lda)
    scores_pls_centroid = np.array(scores_pls_centroid)
    scores_galeano = np.array(scores_galeano)
    scores_rkc = np.array(scores_rkc)

    mean_scores = np.mean(scores, axis=1)
    mean_scores_real_bayes = np.mean(scores_real_bayes, axis=1)
    mean_scores_real_bayes_synt = np.mean(scores_real_bayes_synt, axis=1)
    mean_scores_lda = np.mean(scores_lda, axis=1)
    mean_scores_pls_centroid = np.mean(scores_pls_centroid, axis=1)
    mean_scores_galeano = np.mean(scores_galeano, axis=1)
    mean_scores_rkc = np.mean(scores_rkc, axis=1)
    mean_scores_theoretical = np.array(mean_scores_theoretical)

    std_scores = np.std(scores, axis=1)
    std_scores_real_bayes = np.std(scores_real_bayes, axis=1)
    std_scores_real_bayes_synt = np.std(scores_real_bayes_synt, axis=1)
    std_scores_lda = np.std(scores_lda, axis=1)
    std_scores_pls_centroid = np.std(scores_pls_centroid, axis=1)
    std_scores_galeano = np.std(scores_galeano, axis=1)
    std_scores_rkc = np.std(scores_rkc, axis=1)
    std_scores_theoretical = np.array(std_scores_theoretical)

    legend_scores = 'NP-Rule'
    legend_scores_real_bayes = 'e-QDA'
    legend_scores_real_bayes_synt = 'QDA'
    legend_scores_lda = 'LDA'
    legend_scores_pls_centroid = 'PLS+Centroid'
    legend_scores_galeano = 'PCA+QDA'
    legend_scores_rkc = 'RKC'
    legend_theoretical = 'Theoretical'

    # plt.title('Accuracy')
    plt.figure()
    plot_with_var(mean=mean_scores_theoretical, std=std_scores_theoretical,
                  label=legend_theoretical, color='C4', std_span=2)
    plot_with_var(mean=mean_scores, std=std_scores,
                  label=legend_scores, color='C0', linestyle=':', marker='o')
    plot_with_var(mean=mean_scores_real_bayes_synt,
                  std=std_scores_real_bayes_synt,
                  label=legend_scores_real_bayes_synt, color='C1',
                  linestyle='--', marker='^')
    plot_with_var(mean=mean_scores_real_bayes, std=std_scores_real_bayes,
                  label=legend_scores_real_bayes,
                  color='C2', linestyle='-.', marker='v')
    plot_with_var(mean=mean_scores_lda, std=std_scores_lda,
                  label=legend_scores_lda, color='C3', marker='s')
    plot_with_var(mean=mean_scores_pls_centroid, std=std_scores_pls_centroid,
                  label=legend_scores_pls_centroid, color='C5', marker='X')
    plot_with_var(mean=mean_scores_galeano, std=std_scores_galeano,
                  label=legend_scores_galeano, color='C6', marker='p')
    plot_with_var(mean=mean_scores_rkc, std=std_scores_rkc,
                  label=legend_scores_rkc, color='C7', marker='*')
    plt.xticks(*list(zip(*[(i, 2**i)
                           for i in range(n_points_segment_pow + 1)])))
    plt.xlabel("$N_b$")
    plt.ylabel("Accuracy")
    leg = plt.legend(loc="upper left")
    leg.get_frame().set_alpha(1)

    plt.xlim(0, n_points_segment_pow)
    plt.ylim(top=1.05)
    plt.axhline(1, linestyle=':', color='black')
    if savefig:
        plt.savefig(savefig, bbox_inches="tight", pad_inches=0)
