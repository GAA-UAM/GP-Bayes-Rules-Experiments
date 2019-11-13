from . import experiment
from . import plot
from ..common.classification import classification_test_common
from .brownian_step_classifier import BrownianStepClassifier


@experiment.capture
def classification_test(X_train_list, y_train_list, X_test_list, y_test_list,
                        max_pow, _run):

    classifiers_fd = {
        'optimal': lambda **_: BrownianStepClassifier()
    }

    scores, _ = classification_test_common(
        X_train_list=X_train_list,
        y_train_list=y_train_list,
        X_test_list=X_test_list,
        y_test_list=y_test_list,
        max_pow=max_pow,
        _run=_run,
        additional_classifiers_fd=classifiers_fd)

    plot.plot_scores(max_pow=max_pow,
                     scores=scores,
                     legend_scores_optimal='Step-Rule',
                     _run=_run,
                     optimal_accuracy=1)
