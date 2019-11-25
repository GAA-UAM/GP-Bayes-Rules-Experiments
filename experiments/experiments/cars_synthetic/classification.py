import matplotlib.pyplot as plt

from . import experiment, plot
from ..common.classification import classification_test_common, classifier_qda
from ..common.longvar import LongitudinalVarianceClassifier


@experiment.capture
def classification_test(X_train_list, y_train_list, X_test_list, y_test_list,
                        max_pow, _run, show_plot=False):

    classifiers_fd = {
        'optimal': lambda **_: LongitudinalVarianceClassifier(),
        'brownian_qda': lambda **_: LongitudinalVarianceClassifier(
            real_bayes_rule=True, synthetic_covariance=True)
    }

    classifiers_matrix = {
        'qda': classifier_qda
    }

    scores, _ = classification_test_common(
        X_train_list=X_train_list,
        y_train_list=y_train_list,
        X_test_list=X_test_list,
        y_test_list=y_test_list,
        max_pow=max_pow,
        _run=_run,
        additional_classifiers_fd=classifiers_fd,
        additional_classifiers_matrix=classifiers_matrix,
        start_pow=0,
        no_common_classifiers=True)
