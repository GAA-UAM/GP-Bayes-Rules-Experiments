import scipy.stats
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class BrownianStepClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, significance=0.05):
        self.significance = significance

    def fit(self, X, y):

        # Subtract class 0 mean
        X = X - X[y == 0].mean()

        class_1 = X[y == 1]

        test_0_mean = scipy.stats.ttest_1samp(
            class_1.data_matrix[..., 0], 0, axis=0)
        non_zero_mean = test_0_mean[1] < self.significance

        # It is easier to abandon the null-hypothesis (mean zero) when the
        # mean IS zero than to accept it when the mean IS NOT zero
        # Thus, we find the last zero mean result
        indices_zero = np.argwhere(~non_zero_mean)
        self.index_zero_ = int(indices_zero[-1])
        self.index_nonzero_ = self.index_zero_ + 1

        self.estimated_nonzero_mean_ = np.mean(
            class_1.data_matrix[:, self.index_nonzero_:, 0])

        return self

    def predict(self, X):

        diff = (X.data_matrix[:, self.index_nonzero_, 0]
                - X.data_matrix[:, self.index_zero_, 0])

        return diff > (self.estimated_nonzero_mean_ / 2)
