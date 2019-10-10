import copy

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfda

from .classification import classification_test


# Change for different plots
asset_labels_used = ['TSLA', 'GM']
# asset_labels_used = ['TSLA', 'ETR:BMW']
# asset_labels_used = ['GM', 'ETR:BMW']
n_points_segment_pow = 5


def plot_trajectories(trajectories, print_left=False):
    colors = ['blue', 'red', 'green']

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    lines = []
    labels = []

    for i, key in enumerate(trajectories):
        ls = ax.plot(trajectories[key].data_matrix[..., 0].T,
                     color=colors[i], alpha=0.6)
        lines.append(ls[0])
        labels.append(key)

    dict_names = {
        'TSLA': 'Tesla',
        'GM': 'GM',
        'ETR:BMW': 'BMW'
    }

    ax.legend(lines, [dict_names.get(l, l) for l in labels], loc="upper left")
    ax.set_xlim(0, 2**n_points_segment_pow)
    ax.set_xlabel(r"$t$")
    if print_left:
        ax.set_ylabel("Log returns")

    return fig


def configure_matplotlib():
    plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['text.usetex'] = True


def get_original_data():
    asset_data = pd.read_hdf("/home/carlos/Dropbox/CarsData.h5")

    # Time series of closing prices [in Pyhton, a DataFrame]
    data = asset_data[' Close']
    data = data.dropna()
    assert np.count_nonzero(np.isnan(data.values)) == 0
    data = data.interpolate(limit_direction='both')
    assert data.shape == (1007, 3)

    return data


def compute_log_returns(data):
    dict_series = {}

    for name in data:
        trajs = data[name].values
        trajs = trajs / trajs[0]
        trajs = np.log(trajs)
        dict_series[name] = trajs

    return dict_series


def split_data(data):

    first_key, *_ = data

    # We will select segments of data with a power of two plus one
    # number of points and discard the remaining data
    n_points_segment = 2**n_points_segment_pow + 1
    n_segments = data[first_key].shape[0] // n_points_segment

    dict_subseries = {}

    for key, value in data.items():
        subseries = np.split(value[:n_points_segment * n_segments], n_segments)
        subseries = np.array(subseries)
        dict_subseries[key] = skfda.FDataGrid(
            data_matrix=subseries,
            sample_points=range(subseries.shape[1]))

    return dict_subseries


def transform_to_increments(data):
    # The term log(x_0) can be eliminated by subtracting the value at t=0.
    dict_increments = {}
    for key in data:
        dict_increments[key] = copy.copy(data[key])
        dict_increments[key].data_matrix = (
            data[key].data_matrix - data[key][:, (0,)].data_matrix)

    return dict_increments


def get_real_data():

    original_data = get_original_data()

    # original_data.plot(subplots=True, grid=True, figsize=(8, 6))

    log_returns = compute_log_returns(original_data)

    splitted_data = split_data(log_returns)
    incr_data = transform_to_increments(splitted_data)

    return incr_data


def get_synthetic_data(n_samples, n_features, class_variances):

    class_0 = skfda.datasets.make_gaussian_process(
        n_samples, n_features,
        cov=skfda.misc.covariances.Brownian(variance=class_variances[0]))

    class_1 = skfda.datasets.make_gaussian_process(
        n_samples, n_features,
        cov=skfda.misc.covariances.Brownian(variance=class_variances[1]))

    return {"class_0": class_0, "class_1": class_1}


def compute_class_variances(data):
    class_variances = []

    for _, value in data.items():
        var = skfda.exploratory.stats.var(
            value[:, 1:]) / np.arange(1, value.data_matrix.shape[1])
        class_variances.append(var.data_matrix[..., 0][0, -1])

    return class_variances


def filter_data(data, keys):
    return {k: data[k] for k in keys}


def main():
    configure_matplotlib()

    real_data = get_real_data()
    real_data = filter_data(real_data, asset_labels_used)

    class_variances = compute_class_variances(real_data)

    item, *_ = real_data.values()

    synthetic_data = get_synthetic_data(
        item.n_samples,
        len(item.sample_points[0]),
        class_variances)

    data = real_data

    plot_trajectories(data, asset_labels_used)
    # fig.savefig("/home/carlos/kk2.pdf", bbox_inches="tight", pad_inches=0)

    classification_test(data,
                        n_points_segment_pow=n_points_segment_pow,
                        class_variances=class_variances)

    plt.show()
