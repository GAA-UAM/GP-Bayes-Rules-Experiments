import matplotlib
import matplotlib.pyplot as plt

from ..common.plot import configure_matplotlib, plot_experiments_common
from .data import get_real_data, filter_data, get_asset_labels_used


def get_dict_by_id(id):
    from incense import ExperimentLoader

    loader = ExperimentLoader(
        # None if MongoDB is running on localhost or "mongodb://mongo:27017"
        # when running in devcontainer.
        mongo_uri=None,
        db_name='GPBayes'
    )

    exp = loader.find_by_id(id)

    max_pow = exp.config['max_pow']

    scores = exp.info['scores']
    scores = {key: value for key, value in scores.items() if
              key != 'brownian_qda'}

    theoretical_mean = exp.info['theoretical_mean']
    theoretical_std = exp.info['theoretical_std']

    return {
        'max_pow': max_pow,
        'scores': scores,
        'optimal_accuracy': 1,
        'std_span': 0,
        'theoretical_mean': theoretical_mean,
        'theoretical_std': theoretical_std,
        'start_pow': 0
    }


def plot_trajectories(trajectories, max_pow, print_left=False, axes=None):
    colors = ['blue', 'red', 'green']

    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
    else:
        fig = axes.figure
    lines = []
    labels = []

    for i, key in enumerate(trajectories):
        ls = axes.plot(trajectories[key].data_matrix[..., 0].T,
                       color=colors[i], alpha=0.6)
        lines.append(ls[0])
        labels.append(key)

    dict_names = {
        'TSLA': 'Tesla',
        'GM': 'GM',
        'ETR:BMW': 'BMW'
    }

    axes.legend(lines, [dict_names.get(l, l)
                        for l in labels], loc="lower left", ncol=2)
    axes.set_xlim(0, 2**max_pow)
    axes.set_xlabel(r"$t$")
    if print_left:
        axes.set_ylabel("Log returns")

    return fig


def plot_experiments(ids, titles, data_path, **kwargs):
    from incense import ExperimentLoader

    loader = ExperimentLoader(
        # None if MongoDB is running on localhost or "mongodb://mongo:27017"
        # when running in devcontainer.
        mongo_uri=None,
        db_name='GPBayes'
    )

    configure_matplotlib()

    n_experiments = len(ids)
    default_figsize = matplotlib.rcParams['figure.figsize']

    fig, axes = plt.subplots(2, n_experiments, figsize=(
        default_figsize[0] * n_experiments, default_figsize[1] * 3))

    for i, id in enumerate(ids):
        exp = loader.find_by_id(id)

        max_pow = exp.config['max_pow']

        compare_tesla = exp.config['compare_tesla']
        compare_gm = exp.config['compare_gm']
        compare_bmw = exp.config['compare_bmw']
        asset_labels_used = get_asset_labels_used(
            compare_tesla=compare_tesla,
            compare_gm=compare_gm,
            compare_bmw=compare_bmw)

        real_data = get_real_data(data_path, max_pow)
        real_data = filter_data(real_data, asset_labels_used)

        plot_trajectories(real_data, max_pow,
                          print_left=(i == 0),
                          axes=axes[0, i])
        axes[0, i].set_title(titles[i])

    return plot_experiments_common(ids, get_dict_by_id, axes=axes[1],
                                   top=0.95, bottom=0.15, **kwargs)
