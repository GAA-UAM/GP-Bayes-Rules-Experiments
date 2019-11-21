import matplotlib.pyplot as plt
import numpy as np
import skfda

from . import experiment
from ..common.plot import configure_matplotlib
from .classification import classification_test
from .data import get_real_data, filter_data, get_asset_labels_used


@experiment.config
def config():
    max_pow = 5  # @UnusedVariable
    compare_tesla = False
    compare_gm = False
    compare_bmw = False
    data_path = ""
    assert data_path

    asset_labels_used = get_asset_labels_used(  # @UnusedVariable
        compare_tesla=compare_tesla,
        compare_gm=compare_gm,
        compare_bmw=compare_bmw)

    show_plot = False  # @UnusedVariable


def compute_class_variances(data):
    class_variances = []

    for _, value in data.items():
        var = skfda.exploratory.stats.var(
            value[:, 1:]) / np.arange(1, value.data_matrix.shape[1])
        class_variances.append(var.data_matrix[..., 0][0, -1])

    return class_variances


@experiment.capture
def main(data_path, asset_labels_used, max_pow):
    configure_matplotlib()

    real_data = get_real_data(data_path, max_pow)
    real_data = filter_data(real_data, asset_labels_used)

    class_variances = compute_class_variances(real_data)

    # item, *_ = real_data.values()

    # synthetic_data = get_synthetic_data(
    #    item.n_samples,
    #    len(item.sample_points[0]),
    #    class_variances)

    data = real_data

    # plot_trajectories(data, max_pow, asset_labels_used)
    # fig.savefig("/home/carlos/kk2.pdf", bbox_inches="tight", pad_inches=0)

    classification_test(data,
                        max_pow=max_pow,
                        class_variances=class_variances)

    plt.show()
