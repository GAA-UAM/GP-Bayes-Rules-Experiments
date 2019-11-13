import matplotlib.pyplot as plt

from . import experiment, plot
from ..common.classification import classification_test_common
from ..common.longvar import LongitudinalVarianceClassifier


@experiment.capture
def classification_test(X_train_list, y_train_list, X_test_list, y_test_list,
                        max_pow, _run, show_plot=False):

    classifiers_fd = {
        'optimal': lambda **_: LongitudinalVarianceClassifier(),
        'brownian_qda': lambda **_: LongitudinalVarianceClassifier(
            real_bayes_rule=True, synthetic_covariance=True)
    }

    scores, _ = classification_test_common(
        X_train_list=X_train_list,
        y_train_list=y_train_list,
        X_test_list=X_test_list,
        y_test_list=y_test_list,
        max_pow=max_pow,
        _run=_run,
        additional_classifiers_fd=classifiers_fd)

    if show_plot:
        plot.plot_scores(max_pow=max_pow,
                         scores=scores,
                         legend_scores_optimal='Brownian-Bridge-Rule',
                         _run=None,
                         optimal_accuracy=1)

        plt.show()
