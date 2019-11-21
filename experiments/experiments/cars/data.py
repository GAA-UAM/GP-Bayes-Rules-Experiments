import copy

import numpy as np
import pandas as pd
import skfda


def get_original_data(data_path):
    asset_data = pd.read_hdf(data_path)

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


def split_data(data, max_pow):

    first_key, *_ = data

    # We will select segments of data with a power of two plus one
    # number of points and discard the remaining data
    n_points_segment = 2**max_pow + 1
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


def get_real_data(data_path, max_pow):

    original_data = get_original_data(data_path)

    # original_data.plot(subplots=True, grid=True, figsize=(8, 6))

    log_returns = compute_log_returns(original_data)

    splitted_data = split_data(log_returns, max_pow)
    incr_data = transform_to_increments(splitted_data)

    return incr_data


def get_asset_labels_used(compare_tesla, compare_gm, compare_bmw):
    asset_labels_used = []
    if compare_tesla:
        asset_labels_used.append('TSLA')
    if compare_gm:
        asset_labels_used.append('GM')
    if compare_bmw:
        asset_labels_used.append('ETR:BMW')
    assert len(asset_labels_used) == 2

    return asset_labels_used


def filter_data(data, keys):
    return {k: data[k] for k in keys}
