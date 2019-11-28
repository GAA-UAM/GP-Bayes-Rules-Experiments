import collections.abc
import tempfile

import matplotlib
from matplotlib.ticker import MaxNLocator

import matplotlib.pyplot as plt
import numpy as np


def configure_matplotlib():
    matplotlib.rcParams['axes.titlesize'] = 30
    matplotlib.rcParams['axes.labelsize'] = 25
    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['legend.fontsize'] = 20
    matplotlib.rcParams['xtick.labelsize'] = 16
    matplotlib.rcParams['ytick.labelsize'] = 16
    matplotlib.rcParams['figure.titlesize'] = 30


def plot_with_var(ax, mean, std, color, label, std_span=0, **kwargs):

    for multiple in range(std_span, 0, -1):
        ax.fill_between(range(len(mean)), mean - multiple *
                        std, mean + multiple * std, color=color, alpha=0.15)
    ax.plot(mean, label=label, color=color, **kwargs)


def plot_scores(max_pow, scores, _run, optimal_accuracy,
                std_span=1, plot_y_label="Accuracy", plot_legend=True,
                theoretical_mean=None, theoretical_std=None,
                start_pow=1, ylim_top=1.05, ylim_bottom=None, axes=None):

    configure_matplotlib()

    # plt.title('Accuracy')
    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
    else:
        fig = axes.figure

    mean_scores = {key: np.mean(s, axis=1) for key, s in scores.items()}
    std_scores = {key: np.std(s, axis=1) for key, s in scores.items()}

    legend_scores_optimal = 'Limit-Rule'
    legend_scores_brownian_qda = 'Brownian-QDA'
    legend_scores_lda = 'LDA'
    legend_scores_qda = 'QDA'
    legend_scores_pls_centroid = 'PLS+Centroid'
    legend_scores_pca_qda = 'PCA+QDA'
    legend_scores_rkc = 'RKC'
    legend_theoretical = 'Theoretical'

    if theoretical_mean is not None:
        plot_with_var(axes,
                      mean=theoretical_mean,
                      std=theoretical_std,
                      std_span=2,
                      label=legend_theoretical, color='C4')
    plot_with_var(axes,
                  mean=mean_scores['optimal'], std=std_scores['optimal'],
                  std_span=std_span,
                  label=legend_scores_optimal, color='C0', linestyle=':',
                  marker='o')
    if 'brownian_qda' in scores:
        plot_with_var(axes,
                      mean=mean_scores['brownian_qda'],
                      std=std_scores['brownian_qda'],
                      label=legend_scores_brownian_qda,
                      std_span=std_span,
                      color='C1', linestyle='--', marker='^')
    if 'qda' in scores:
        plot_with_var(axes,
                      mean=mean_scores['qda'], std=std_scores['qda'],
                      label=legend_scores_qda,
                      std_span=std_span,
                      color='C2', linestyle='-.', marker='v')
    if 'pca_qda' in scores:
        plot_with_var(axes,
                      mean=mean_scores['pca_qda'], std=std_scores['pca_qda'],
                      std_span=std_span,
                      label=legend_scores_pca_qda, color='C6', marker='X')
    if 'lda' in scores:
        plot_with_var(axes,
                      mean=mean_scores['lda'], std=std_scores['lda'],
                      std_span=std_span,
                      label=legend_scores_lda, color='C3',
                      linestyle='--', marker='s')
    if 'pls_centroid' in scores:
        plot_with_var(axes,
                      mean=mean_scores['pls_centroid'],
                      std=std_scores['pls_centroid'],
                      std_span=std_span,
                      label=legend_scores_pls_centroid,
                      color='C5', linestyle='-.', marker='p')
    if 'rkc' in scores:
        plot_with_var(axes,
                      mean=mean_scores['rkc'], std=std_scores['rkc'],
                      std_span=std_span,
                      label=legend_scores_rkc, color='C7', marker='*')
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes.set_xticklabels([2**i for i in range(start_pow, max_pow + 1)])
    axes.set_xlabel("$N_b$")
    if plot_y_label:
        axes.set_ylabel(plot_y_label)

    if plot_legend:
        leg = axes.legend(loc="best", fontsize=12)
        leg.get_frame().set_alpha(1)

    axes.set_xlim(0, max_pow - start_pow)
    if ylim_top is not None:
        axes.set_ylim(top=ylim_top)
    if ylim_bottom is not None:
        axes.set_ylim(bottom=ylim_bottom)
    axes.axhline(optimal_accuracy, linestyle=':', color='black')

    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmpfile:
        fig.savefig(tmpfile, format="pdf")
        if _run is not None:
            _run.add_artifact(tmpfile.name, name="plot.pdf")

    return fig


def plot_experiments_common(ids, function, titles=None, title=None, axes=None,
                            top=None, bottom=0.20):
    configure_matplotlib()

    n_experiments = len(ids)
    default_figsize = matplotlib.rcParams['figure.figsize']

    if axes is None:
        fig, axes = plt.subplots(1, n_experiments, figsize=(
            default_figsize[0] * n_experiments, default_figsize[1] * 1.5))
    else:
        fig = axes[0].figure

    y_bottoms = []

    for i, id in enumerate(ids):
        options = function(id)

        if i != 0:
            options['plot_y_label'] = False

        fig = plot_scores(**options,
                          _run=None,
                          plot_legend=False,
                          axes=axes[i])

        y_bottoms.append(axes[i].get_ylim()[0])

        if titles is not None:
            axes[i].set_title(titles[i])

    for i, id in enumerate(ids):
        axes[i].set_ylim(bottom=np.min(y_bottoms))

    if title is not None:
        fig.set_title(title)

    fig.tight_layout()
    fig.subplots_adjust(top=top, bottom=bottom)
    handles, labels = axes[0].get_legend_handles_labels()

    # Theoretical at the end
    if 'Theoretical' in labels:
        pos = labels.index('Theoretical')

        handles = handles[:pos] + handles[pos + 1:] + [handles[pos]]
        labels = labels[:pos] + labels[pos + 1:] + [labels[pos]]

    leg = fig.legend(handles, labels, loc="lower center",
                     bbox_to_anchor=(0.5, 0),
                     bbox_transform=fig.transFigure, ncol=7)
    leg.get_frame().set_alpha(1)

    return fig


def get_confusion_matrix_stat(confusion_matrices, stat):

    result = {}

    for key, value in confusion_matrices.items():

        result[key] = []

        for pow_list in value:

            if isinstance(pow_list, collections.abc.Iterable):
                new_pow_list = [stat(repetition) for repetition in pow_list]
            else:
                new_pow_list = pow_list

            result[key].append(new_pow_list)

        result[key] = np.array(result[key][1:], ndmin=2)

    return result


def fdr_stat(confusion_matrix):
    return confusion_matrix[0, 1] / (
        confusion_matrix[0, 0] + confusion_matrix[0, 1])


def for_stat(confusion_matrix):
    return confusion_matrix[1, 0] / (
        confusion_matrix[1, 0] + confusion_matrix[1, 1])


def plot_confusion_matrix_stat(id, stat, title=None, plot_y_label=True, ylim_top=None):
    from incense import ExperimentLoader

    loader = ExperimentLoader(
        # None if MongoDB is running on localhost or "mongodb://mongo:27017"
        # when running in devcontainer.
        mongo_uri=None,
        db_name='GPBayes'
    )

    exp = loader.find_by_id(id)

    max_pow = exp.config['max_pow']

    confusion_matrices = exp.info['confusion_matrices']

    title = exp.experiment.name

    titles_dict = {
        'brownian_step': 'Brownian step example',
        'brownian_bridge': 'Brownian bridge example',
        'brownian_variances': 'Brownian variances example',
        'cars': 'Cars experiment'
    }

    stat_dict = get_confusion_matrix_stat(confusion_matrices, stat)

    fig = plot_scores(max_pow=max_pow,
                      scores=stat_dict,
                      _run=None,
                      optimal_accuracy=0,
                      plot_y_label=plot_y_label,
                      ylim_top=ylim_top)

    fig.axes[0].set_title(titles_dict[title])

    return fig


def plot_confusion_matrix(id, n_samples, ylim_top=None,
                          optimal_accuracy=[1, 0, 0, 1]):
    from incense import ExperimentLoader

    configure_matplotlib()

    loader = ExperimentLoader(
        # None if MongoDB is running on localhost or "mongodb://mongo:27017"
        # when running in devcontainer.
        mongo_uri=None,
        db_name='GPBayes'
    )

    exp = loader.find_by_id(id)

    max_pow = exp.config['max_pow']

    confusion_matrices = exp.info['confusion_matrices']

    confusion_matrices = {key: value for key, value
                          in confusion_matrices.items() if
                          key != 'brownian_qda'}

    title = exp.experiment.name

    titles_dict = {
        'brownian_step': 'Brownian step example',
        'brownian_bridge': 'Brownian bridge example',
        'brownian_variances': 'Brownian variances example',
        'cars': 'Cars experiment'
    }

    default_figsize = matplotlib.rcParams['figure.figsize']

    fig, axes = plt.subplots(2, 2, figsize=(
        default_figsize[0] * 2.2, default_figsize[1] * 3))

    true_pos = get_confusion_matrix_stat(confusion_matrices,
                                         lambda x: x[0, 0])
    false_pos = get_confusion_matrix_stat(confusion_matrices,
                                          lambda x: x[0, 1])
    false_neg = get_confusion_matrix_stat(confusion_matrices,
                                          lambda x: x[1, 0])
    true_neg = get_confusion_matrix_stat(confusion_matrices,
                                         lambda x: x[1, 1])

    for scores, index, optimal in zip(
        [true_pos, false_pos, false_neg, true_neg],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
            optimal_accuracy):
        plot_scores(max_pow=max_pow,
                    scores=scores,
                    _run=None,
                    optimal_accuracy=optimal * n_samples // 2,
                    plot_y_label=False,
                    ylim_top=ylim_top,
                    ylim_bottom=0,
                    plot_legend=False,
                    axes=axes[index])

    axes[0, 0].set_xlabel(None)
    axes[0, 1].set_xlabel(None)

    fig.suptitle(titles_dict[title])

    fig.tight_layout()
    fig.subplots_adjust(top=0.93, bottom=0.15, hspace=0.1)
    handles, labels = axes[0, 0].get_legend_handles_labels()

    leg = fig.legend(handles, labels, loc="lower center",
                     bbox_to_anchor=(0.5, 0),
                     bbox_transform=fig.transFigure, ncol=7)
    leg.get_frame().set_alpha(1)

    return fig
