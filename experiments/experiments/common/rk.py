'''
Created on 27 jun. 2017

@author: carlos
'''

import sklearn.utils.validation

import numpy as np
import numpy.linalg as linalg


def rk(X, Y, n_components: int=1):
    '''
    Parameters
    ----------
    X
        Matrix of trajectories
    Y
        Vector of class labels
    n_components
        Number of selected components
    '''

    X = np.atleast_2d(X)
    assert n_components >= 1
    assert n_components <= X.shape[1]

    Y = np.asarray(Y)

    selected_features = np.zeros(n_components, dtype=int)
    score = np.zeros(n_components)
    indexes = np.arange(0, X.shape[1])

    # Calculate means and covariance matrix
    class_1_trajectories = X[Y.ravel() == 1]
    class_0_trajectories = X[Y.ravel() == 0]

    means = (np.mean(class_1_trajectories, axis=0) -
             np.mean(class_0_trajectories, axis=0))

    class_1_count = sum(Y)
    class_0_count = Y.shape[0] - class_1_count

    class_1_proportion = class_1_count / Y.shape[0]
    class_0_proportion = class_0_count / Y.shape[0]

    # The result should be casted to 2D because of bug #11502 in numpy
    variances = (
        class_1_proportion * np.atleast_2d(
            np.cov(class_1_trajectories, rowvar=False, bias=True)) +
        class_0_proportion * np.atleast_2d(
            np.cov(class_0_trajectories, rowvar=False, bias=True)))

    # The first variable maximizes |mu(t)|/sigma(t)
    mu_sigma = np.abs(means) / np.sqrt(np.diag(variances))

    selected_features[0] = np.argmax(mu_sigma)
    score[0] = mu_sigma[selected_features[0]]
    indexes = np.delete(indexes, selected_features[0])

    for i in range(1, n_components):
        aux = np.zeros_like(indexes, dtype=np.float_)

        for j in range(0, indexes.shape[0]):
            new_selection = np.concatenate([selected_features[0:i],
                                            [indexes[j]]])

            new_means = np.atleast_2d(means[new_selection])

            lstsq_solution = linalg.lstsq(
                variances[new_selection[:, np.newaxis], new_selection],
                new_means.T, rcond=None)[0]

            aux[j] = new_means @ lstsq_solution

        aux2 = np.argmax(aux)
        selected_features[i] = indexes[aux2]
        score[i] = aux[aux2]
        indexes = np.delete(indexes, aux2)

    return selected_features, score


class RK(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    '''
    Sklearn transformer that uses the RK method.
    '''

    def __init__(self,
                 n_components: int=1):
        self.n_components = n_components

    def fit(self, X, y):

        X, y = sklearn.utils.validation.check_X_y(X, y)

        self.features_shape_ = X.shape[1:]

        self.results_ = rk(
            X=X,
            Y=y,
            n_components=self.n_components)

        return self

    def transform(self, X, Y=None):

        sklearn.utils.validation.check_is_fitted(self, ['features_shape_',
                                                        'results_'])

        X = sklearn.utils.validation.check_array(X)

        if X.shape[1:] != self.features_shape_:
            raise ValueError("The trajectories have a different number of "
                             "points than the ones fitted")

        return X[:, self.results_[0]]

    def get_support(self, indices: bool=False):
        indexes_unraveled = self.results_[0]
        if indices:
            return indexes_unraveled
        else:
            mask = np.zeros(self.features_shape_[0], dtype=bool)
            mask[self.results_[0]] = True
            return mask
