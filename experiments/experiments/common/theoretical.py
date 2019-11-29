import scipy.special
import numpy as np


class TheoreticalBounds():

    def __init__(self, variance_0, variance_1, n_points, mean_0=0, mean_1=0,
                 prior_0=0.5, prior_1=0.5):
        self.variance = [None, None]
        self.variance[0] = variance_0
        self.variance[1] = variance_1

        self.n_points = n_points

        sample_points = np.arange(self.n_points)

        base_cov = np.minimum(sample_points[:, None], sample_points)

        self.covariance_matrix = [None, None]
        self.covariance_matrix[0] = self.variance[0] * base_cov
        self.covariance_matrix[1] = self.variance[1] * base_cov

        self.mean = [None, None]
        self.mean[0] = mean_0 + np.zeros(n_points)
        self.mean[1] = mean_1 + np.zeros(n_points)

        self._discard_first()

        inv = np.linalg.inv

        self.covariance_matrix_inv = [None, None]
        self.covariance_matrix_inv[0] = inv(self.covariance_matrix[0])
        self.covariance_matrix_inv[1] = inv(self.covariance_matrix[1])

        self.prior = [None, None]
        self.prior[0] = prior_0
        self.prior[1] = prior_1

    def _discard_first(self):
        self.covariance_matrix[0] = self.covariance_matrix[0][1:, 1:]
        self.covariance_matrix[1] = self.covariance_matrix[1][1:, 1:]

        self.mean[0] = self.mean[0][1:]
        self.mean[1] = self.mean[1][1:]

    def log_k(self, i):
        return np.sum(np.log(np.linalg.eigh(self.covariance_matrix[i])[0]))

    def mu(self, i):
        other = int(not i)
        identity = np.identity(self.covariance_matrix[i].shape[0])

        cov = self.covariance_matrix
        cov_inv = self.covariance_matrix_inv

        m = self.mean[i] - self.mean[other]

        return ((-1)**i * 0.5 * np.trace(identity - cov_inv[other] @ cov[i]) +
                (-1)**other * 0.5 * m @ cov_inv[other] @ m[:, None] +
                0.5 * (self.log_k(0) - self.log_k(1)) +
                np.log(self.prior[1] / self.prior[0]))

    def s_sqr(self, i):
        other = int(not i)
        identity = np.identity(self.covariance_matrix[i].shape[0])

        cov = self.covariance_matrix
        cov_inv = self.covariance_matrix_inv

        mat = identity - cov_inv[other] @ cov[i]

        m = self.mean[i] - self.mean[other]

        return (0.5 * np.trace(mat @ mat) +
                m @ cov_inv[other] @ cov[i] @ cov_inv[other] @ m[:, None])

    def s(self, i):
        return np.sqrt(self.s_sqr(i))

    def error_probability_with_priors(self, priors):
        return (0.5
                - 0.5 *
                priors[0] *
                scipy.special.erf(-self.mu(0) / (self.s(0) * np.sqrt(2)))
                - 0.5 *
                priors[1] *
                scipy.special.erf(self.mu(1) / (self.s(1) * np.sqrt(2)))
                )

    def error_probability(self):
        return self.error_probability_with_priors(self.prior)

    def error_probability_std(self, n_traj):
        e_prob = self.error_probability()

        return np.sqrt(e_prob * (1 - e_prob) / n_traj)
