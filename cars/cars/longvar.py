from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np
import skfda


def fda_longitudinal_variance_terms(logfda):
    d_subs = logfda.data_matrix[..., 0][:, 1:] - \
        logfda.data_matrix[..., 0][:, :-1]
    t_subs = np.array(
        logfda.sample_points[0][1:]) - np.array(logfda.sample_points[0][:-1])
    terms = d_subs / np.sqrt(t_subs)

    return terms**2


def fda_longitudinal_variance(logfda):
    terms = fda_longitudinal_variance_terms(logfda)
    variance = np.nanmean(terms, axis=1)
    return variance


class LongitudinalVarianceClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, simple_difference_comparison=False,
                 real_bayes_rule=False,
                 class_variances=None,
                 synthetic_covariance=False,
                 longitudinal_variances=True,
                 linear=False):
        self.simple_difference_comparison = simple_difference_comparison
        self.class_variances = class_variances
        self.real_bayes_rule = real_bayes_rule
        self.synthetic_covariance = synthetic_covariance
        self.longitudinal_variances = longitudinal_variances
        self.linear = linear

    def fit(self, X, y):

        # Check that X and y have correct shape
        _, y = check_X_y(X.data_matrix[..., 0], y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.class_variances_ = (np.zeros(len(self.classes_)) if
                                 self.class_variances is None else
                                 self.class_variances)

        # Data for the exact Bayes rule
        self.class_covariance_matrices_ = [None] * len(self.classes_)
        self.class_mean_ = [None] * len(self.classes_)

        for cl in self.classes_:

            if self.class_variances is None:
                if self.longitudinal_variances:
                    self.class_variances_[cl] = np.mean(
                        fda_longitudinal_variance(X[y == cl]))
                else:
                    var = skfda.exploratory.stats.var(
                        X[y == cl, 1:]) / X.sample_points[1:]
                    self.class_variances_[cl] = var.data_matrix[..., 0][0, -1]

            self.synthetic_matrix = np.minimum(
                X.sample_points[0][:, None], X.sample_points[0])

            if self.synthetic_covariance:
                self.class_covariance_matrices_[
                    cl] = self.class_variances_[cl] * self.synthetic_matrix
            else:
                self.class_covariance_matrices_[cl] = np.cov(
                    X[y == cl].data_matrix[..., 0], rowvar=False)
            self.class_mean_[cl] = skfda.exploratory.stats.mean(X[y == cl])

        self.m_ = self.class_mean_[1] - self.class_mean_[0]

        # Return the classifier
        return self

    def predict(self, X):
        def scalar_product(cl, x, y):
            try:
                cov_inv = np.linalg.inv(
                    self.class_covariance_matrices_[cl][1:, 1:])
            except np.linalg.LinAlgError:
                print(f'Singular matrix with points {x.sample_points}')
                return np.nan

            y = y.data_matrix[..., 0][:, 1:].T
            x = x.data_matrix[..., 0][:, 1:]

            return np.diag(x @ cov_inv @ y)[:, None]

        def logterm():
            if True:  # self.synthetic_covariance:
                mat = self.synthetic_matrix[1:, 1:]
                dim = mat.shape[0]
                det = np.linalg.det(mat)
                det1 = self.class_variances_[1]**dim * det
                det0 = self.class_variances_[0]**dim * det
            else:
                det1 = np.linalg.det(
                    self.class_covariance_matrices_[1][1:, 1:])
                det0 = np.linalg.det(
                    self.class_covariance_matrices_[0][1:, 1:])

            return -0.5 * (np.log(det1 / det0))

        def bayes_rule(x):
            x -= self.class_mean_[0]  # Subtract first class mean

            s1 = -0.5 * (scalar_product(1, x, x) - scalar_product(0, x, x))
            if self.linear:
                s1 *= 0
            s2 = scalar_product(1, x, self.m_)
            s3 = -0.5 * scalar_product(1, self.m_, self.m_)
            s4 = logterm()

            bayes_res = ((s1 + s2 + s3 + s4) > 0).astype(int)

            return bayes_res

        # Check is fit had been called
        check_is_fitted(self, ['class_variances_'])

        # Input validation
        check_array(X.data_matrix[..., 0])

        if self.real_bayes_rule:
            return bayes_rule(X)
        else:
            var = fda_longitudinal_variance(X)

            if self.simple_difference_comparison:
                cl = np.argmin(
                    np.abs(var[:, np.newaxis] - self.class_variances_), axis=1)
            else:
                cl = np.argmin(
                    var[:, np.newaxis] / self.class_variances_
                    + np.log(self.class_variances_), axis=1)

            return cl
