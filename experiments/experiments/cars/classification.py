import matplotlib.pyplot as plt
import numpy as np
import skfda

from . import experiment
from ..common import plot
from ..common.classification import classification_test_common
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


@experiment.capture
def classification_test(data, max_pow,
                        class_variances,
                        _run, show_plot=False):

    X, y = build_dataset(data)

    _run.info["class_variances"] = class_variances

    classifiers_fd = {
        'optimal': lambda **_: LongitudinalVarianceClassifier(),
        'brownian_qda': lambda **_: LongitudinalVarianceClassifier(
            real_bayes_rule=True, synthetic_covariance=True)
    }

    mean_scores_theoretical = [None] * (max_pow + 1)
    std_scores_theoretical = [None] * (max_pow + 1)

    scores, _ = classification_test_common(
        X_train_list=[X],
        y_train_list=[y],
        max_pow=max_pow,
        _run=_run,
        additional_classifiers_fd=classifiers_fd,
        start_pow=0)

    for resolution in range(max_pow + 1):
        theoretical_bounds = TheoreticalBounds(variance_0=class_variances[0],
                                               variance_1=class_variances[1],
                                               mean_0=0, mean_1=0,
                                               n_points=2**resolution + 1)
        mean_scores_theoretical[resolution] = float(
            1 - theoretical_bounds.error_probability())
        std_scores_theoretical[resolution] = float(
            theoretical_bounds.error_probability_std(n_traj=X.shape[0]))

    theoretical_mean = np.array(mean_scores_theoretical)
    theoretical_std = np.array(std_scores_theoretical)

    _run.info["theoretical_mean"] = theoretical_mean
    _run.info["theoretical_std"] = theoretical_std

    if show_plot:
        plot.plot_scores(max_pow=max_pow,
                         scores=scores,
                         legend_scores_optimal='NP-Rule',
                         _run=None,
                         optimal_accuracy=1,
                         std_span=0,
                         theoretical_mean=theoretical_mean,
                         theoretical_std=theoretical_std,
                         start_pow=0)
        plt.show()
