from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np


class BrownianBridgeClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        pass

    def fit(self, X, y):

        return self

    def predict(self, X):

        end_diff = 1 - X.sample_points[0][-1]
        first_term = -0.5 * np.log(end_diff)
        second_term = -0.5 * X.data_matrix[:, -1, 0]**2 / end_diff

        return first_term + second_term > 0
