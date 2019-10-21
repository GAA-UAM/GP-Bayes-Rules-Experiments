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

    def __init__(self,
                 real_bayes_rule=False,
                 class_variances=None,
                 synthetic_covariance=False,
                 longitudinal_variances=True,
                 linear=False):
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

    def decision_function(self, X):
        def _scalar_product_aux(cl, x, y):
            try:
                cov_inv = np.linalg.inv(
                    self.class_covariance_matrices_[cl][1:, 1:])
            except np.linalg.LinAlgError:
                print(f'Singular matrix with points {x.sample_points}')
                return np.nan

            y = y.data_matrix[..., 0][:, 1:].T
            x = x.data_matrix[..., 0][:, 1:]

            return x @ cov_inv @ y

        def scalar_product_quadratic(cl, x, y):
            res = _scalar_product_aux(cl, x, y)

            return np.diag(res)[:, None]

        def scalar_product_mean(cl, x, y):
            return _scalar_product_aux(cl, x, y)

        def logterm():
            if self.synthetic_covariance:
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

        X -= self.class_mean_[0]  # Subtract first class mean

        s1 = -0.5 * (scalar_product_quadratic(1, X, X) -
                     scalar_product_quadratic(0, X, X))
        if self.linear:
            s1 *= 0
        s2 = scalar_product_mean(1, X, self.m_)
        s3 = -0.5 * scalar_product_mean(1, self.m_, self.m_)
        s4 = logterm()

        self.decision_steps_ = (s1, s2, s3, s4)

        return s1 + s2 + s3 + s4

    def predict(self, X):

        def bayes_rule(x):
            bayes_res = (self.decision_function(x) > 0).astype(int)

            return bayes_res

        # Check is fit had been called
        check_is_fitted(self, ['class_variances_'])

        # Input validation
        check_array(X.data_matrix[..., 0])

        if self.real_bayes_rule:
            return bayes_rule(X)
        else:
            var = fda_longitudinal_variance(X)

            cl = np.argmin(
                var[:, np.newaxis] / self.class_variances_
                + np.log(self.class_variances_), axis=1)

            return cl
