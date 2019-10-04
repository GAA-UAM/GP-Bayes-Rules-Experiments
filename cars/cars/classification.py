import sklearn
import sklearn.discriminant_analysis
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import numpy as np
import skfda

from .longvar import LongitudinalVarianceClassifier
from .theoretical import TheoreticalBounds


def compute_class_variances(data):
    class_variances = []

    for _, value in data.items():
        var = skfda.exploratory.stats.var(
            value[:, 1:]) / np.arange(1, value.data_matrix.shape[1])
        class_variances.append(var.data_matrix[..., 0][0, -1])

    return class_variances


def build_dataset(d):
    X = []
    y = []

    for i, (_, value) in enumerate(d.items()):

        X.append(value)
        y.append(np.full(shape=value.shape[0], fill_value=i))

    data_matrix = np.vstack(x.data_matrix for x in X)
    X = skfda.FDataGrid(data_matrix=data_matrix,
                        sample_points=X[0].sample_points)

    y = np.concatenate(y)

    return X, y


def array_with_resolution(array, resolution):
    step = (array.data_matrix.shape[1] - 1) // 2**resolution

    return array[:, ::step]


def dict_with_resolution(d, resolution):
    d_new = {}

    for key, value in d.items():
        d_new[key] = array_with_resolution(value, resolution)

    return d_new


def plot_with_var(mean, std, color, label, std_span=0, **kwargs):

    for multiple in range(std_span, 0, -1):
        plt.fill_between(range(len(mean)), mean - multiple *
                         std, mean + multiple * std, color=color, alpha=0.15)
    plt.plot(mean, label=label, color=color, **kwargs)


def filter_data(data, keys):
    return {k: data[k] for k in keys}


def classification_test(data, n_points_segment_pow, keys, savefig=None):

    data = filter_data(data, keys)

    # Leave one-out
    cv = 10

    scores = [None] * (n_points_segment_pow + 1)
    scores_real_bayes = [None] * (n_points_segment_pow + 1)
    scores_real_bayes_synt = [None] * (n_points_segment_pow + 1)
    scores_lda = [None] * (n_points_segment_pow + 1)
    scores_linear = [None] * (n_points_segment_pow + 1)
    mean_scores_theoretical = [None] * (n_points_segment_pow + 1)
    std_scores_theoretical = [None] * (n_points_segment_pow + 1)

    for resolution in range(n_points_segment_pow + 1):
        d = dict_with_resolution(data, resolution)
        X, y = build_dataset(d)

        clf = LongitudinalVarianceClassifier()
        clf_real_bayes = LongitudinalVarianceClassifier(real_bayes_rule=True)
        clf_real_bayes_synt = LongitudinalVarianceClassifier(
            real_bayes_rule=True, synthetic_covariance=True)
        clf_lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(
            priors=[.5, .5])
        clf_linear = LongitudinalVarianceClassifier(real_bayes_rule=True,
                                                    synthetic_covariance=False,
                                                    linear=True)

        scores[resolution] = cross_val_score(clf, X, y, cv=cv)
        scores_real_bayes[resolution] = cross_val_score(
            clf_real_bayes, X, y, cv=cv)
        scores_real_bayes_synt[resolution] = cross_val_score(
            clf_real_bayes_synt, X, y, cv=cv)
        scores_lda[resolution] = cross_val_score(
            clf_lda, X.data_matrix[..., 0][:, 1:], y, cv=cv)
        scores_linear[resolution] = cross_val_score(clf_linear, X, y, cv=cv)

        mean_0 = 0
        mean_1 = 0

        class_variances = compute_class_variances(data)

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
    scores_linear = np.array(scores_linear)

    mean_scores = np.mean(scores, axis=1)
    mean_scores_real_bayes = np.mean(scores_real_bayes, axis=1)
    mean_scores_real_bayes_synt = np.mean(scores_real_bayes_synt, axis=1)
    mean_scores_lda = np.mean(scores_lda, axis=1)
    mean_scores_theoretical = np.array(mean_scores_theoretical)

    std_scores = np.std(scores, axis=1)
    std_scores_real_bayes = np.std(scores_real_bayes, axis=1)
    std_scores_real_bayes_synt = np.std(scores_real_bayes_synt, axis=1)
    std_scores_lda = np.std(scores_lda, axis=1)
    std_scores_theoretical = np.array(std_scores_theoretical)

    legend_scores = 'NP-Rule'
    legend_scores_real_bayes = 'e-QDA'
    legend_scores_real_bayes_synt = 'QDA'
    legend_scores_lda = 'LDA'
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
    plt.xticks(*list(zip(*[(i, 2**i)
                           for i in range(n_points_segment_pow + 1)])))
    plt.xlabel("$N_b$")
    plt.ylabel("Accuracy")
    leg = plt.legend(loc="upper left")
    leg.get_frame().set_alpha(1)

    plt.xlim(0, n_points_segment_pow)
    plt.ylim(ymax=1.05)
    plt.axhline(1, linestyle=':', color='black')
    if savefig:
        plt.savefig(savefig, bbox_inches="tight", pad_inches=0)
