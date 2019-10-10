import matplotlib.pyplot as plt
import numpy as np
import skfda
from .brownian_step_classifier import BrownianStepClassifier


def generate_data(n_samples=1000, n_features=2**10,
                  step_position=0.5, step_height=1):
    X = skfda.datasets.make_gaussian_process(
        n_samples=n_samples, n_features=n_features)

    X[n_samples // 2:].data_matrix[
        :, X.sample_points[0] > step_position] += step_height

    y = np.zeros(n_samples, dtype=int)
    y[n_samples // 2:] += 1

    return X, y


def main():
    X, y = generate_data()

    X.plot(sample_labels=y)

    cl = BrownianStepClassifier()
    cl.fit(X, y)
    print(cl.predict(X))

    plt.show()
