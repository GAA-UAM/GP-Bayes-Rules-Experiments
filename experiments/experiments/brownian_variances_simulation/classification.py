import numpy as np

from . import experiment
from ..common.classification import classification_test_common
from ..common.longvar import LongitudinalVarianceClassifier
from ..common.theoretical import TheoreticalBounds


@experiment.capture
def classification_test(X_train_list, y_train_list, X_test_list, y_test_list,
                        max_pow, _run, class0_var, class1_var,
                        show_plot=False):

    classifiers_fd = {
        'optimal': lambda **_: LongitudinalVarianceClassifier(
            class_variances=np.array([class0_var, class1_var]))
    }

    mean_scores_theoretical = [None] * (max_pow + 1)
    std_scores_theoretical = [None] * (max_pow + 1)

    scores, _ = classification_test_common(
        X_train_list=X_train_list,
        y_train_list=y_train_list,
        X_test_list=X_test_list,
        y_test_list=y_test_list,
        max_pow=max_pow,
        _run=_run,
        additional_classifiers_fd=classifiers_fd,
        no_common_classifiers=True,
        start_pow=0)

    for resolution in range(max_pow + 1):
        theoretical_bounds = TheoreticalBounds(variance_0=class0_var,
                                               variance_1=class1_var,
                                               mean_0=0, mean_1=0,
                                               n_points=2**resolution + 1)
        mean_scores_theoretical[resolution] = float(
            1 - theoretical_bounds.error_probability())
        std_scores_theoretical[resolution] = float(
            theoretical_bounds.error_probability_std(
                n_traj=X_test_list[0].shape[0]))

    theoretical_mean = np.array(mean_scores_theoretical)
    theoretical_std = np.array(std_scores_theoretical)

    _run.info["theoretical_mean"] = theoretical_mean
    _run.info["theoretical_std"] = theoretical_std
