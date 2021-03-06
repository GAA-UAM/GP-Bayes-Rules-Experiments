import scipy.stats
from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np


class BrownianStepClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        pass

    def fit(self, X, y):

        class_1 = X[y == 1]
        diff = class_1.data_matrix[:, 1:, 0] - class_1.data_matrix[:, :-1, 0]

        test_0_mean = scipy.stats.ttest_1samp(
            diff, 0, axis=0)

        self.index_zero_ = np.argmin(test_0_mean[1])
        self.index_nonzero_ = self.index_zero_ + 1

        self.estimated_nonzero_mean_ = np.mean(
            class_1.data_matrix[:, self.index_nonzero_, 0])

        return self

    def predict(self, X):

        diff = (X.data_matrix[:, self.index_nonzero_, 0]
                - X.data_matrix[:, self.index_zero_, 0])

        return diff > (self.estimated_nonzero_mean_ / 2)
