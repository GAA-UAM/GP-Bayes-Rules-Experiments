import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import skfda

from . import experiment
from .classification import classification_test


@experiment.config
def config():
    max_pow = 10  # @UnusedVariable
    n_tests = 100  # @UnusedVariable
    step_height = 0.3  # @UnusedVariable
    train_n_samples = 1000  # @UnusedVariable
    test_n_samples = 1000  # @UnusedVariable

    random_state_train_seed = 0  # @UnusedVariable
    random_state_test_seed = 1  # @UnusedVariable


def generate_data(n_samples=1000, n_features=2**10 + 1,
                  step_position=0.5, step_height=1, random_state=None):
    X = skfda.datasets.make_gaussian_process(
        n_samples=n_samples, n_features=n_features, random_state=random_state)

    X[n_samples // 2:].data_matrix[
        :, X.sample_points[0] > step_position] += step_height

    y = np.zeros(n_samples, dtype=int)
    y[n_samples // 2:] += 1

    return X, y


def configure_matplotlib():
    plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['text.usetex'] = True


@experiment.capture
def main(max_pow, n_tests, step_height, train_n_samples, test_n_samples,
         random_state_train_seed, random_state_test_seed):

    random_state = np.random.RandomState(random_state_train_seed)
    random_state_test = np.random.RandomState(random_state_test_seed)

    configure_matplotlib()

    X_train_list, y_train_list = zip(*[generate_data(
        n_samples=train_n_samples,
        n_features=2**max_pow + 1,
        step_height=step_height,
        random_state=random_state)
        for _ in range(n_tests)])
    X_test_list, y_test_list = zip(*[generate_data(
        n_samples=test_n_samples,
        n_features=2**max_pow + 1,
        step_height=step_height,
        random_state=random_state_test)
        for _ in range(n_tests)])

    X_train_list[0].plot(sample_labels=y_train_list[0])

    classification_test(X_train_list, y_train_list,
                        X_test_list, y_test_list, max_pow)
