import matplotlib.pyplot as plt
import numpy as np
import skfda
from .classification import classification_test


def generate_data(n_samples=1000, n_features=2**10 + 1,
                  step_position=0.5, step_height=1, random_state=None):
    X = skfda.datasets.make_gaussian_process(
        n_samples=n_samples, n_features=n_features, random_state=random_state)

    X[n_samples // 2:].data_matrix[
        :, X.sample_points[0] > step_position] += step_height

    y = np.zeros(n_samples, dtype=int)
    y[n_samples // 2:] += 1

    return X, y


def main():

    max_pow = 10
    n_tests = 10
    step_height = 0.3
    train_n_samples = 100
    test_n_samples = 1000

    random_state = np.random.RandomState(0)
    random_state_test = np.random.RandomState(1)

    X_train_list, y_train_list = zip(*[generate_data(n_samples=train_n_samples,
                                                     n_features=2**max_pow + 1,
                                                     step_height=step_height,
                                                     random_state=random_state)
                                       for _ in range(n_tests)])
    X_test_list, y_test_list = zip(*[generate_data(n_samples=test_n_samples,
                                                   n_features=2**max_pow + 1,
                                                   step_height=step_height,
                                                   random_state=random_state_test)
                                     for _ in range(n_tests)])

    X_train_list[0].plot(sample_labels=y_train_list[0])

    classification_test(X_train_list, y_train_list,
                        X_test_list, y_test_list, max_pow)

    plt.show()
