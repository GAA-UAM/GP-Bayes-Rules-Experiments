import collections.abc
import tempfile

import matplotlib

import matplotlib.pyplot as plt
import numpy as np


def configure_matplotlib():
    matplotlib.rcParams['axes.titlesize'] = 25
    matplotlib.rcParams['axes.labelsize'] = 20
    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['legend.fontsize'] = 20


def plot_with_var(ax, mean, std, color, label, std_span=0, **kwargs):

    for multiple in range(std_span, 0, -1):
        ax.fill_between(range(len(mean)), mean - multiple *
                        std, mean + multiple * std, color=color, alpha=0.15)
    ax.plot(mean, label=label, color=color, **kwargs)


def plot_scores(max_pow, scores, legend_scores_optimal, _run, optimal_accuracy,
                std_span=1, plot_y_label=True, plot_legend=True,
                theoretical_mean=None, theoretical_std=None,
                start_pow=1, ylim_bottom=None, axes=None):

    configure_matplotlib()

    # plt.title('Accuracy')
    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
    else:
        fig = axes.figure

    mean_scores = {key: np.mean(s, axis=1) for key, s in scores.items()}
    std_scores = {key: np.std(s, axis=1) for key, s in scores.items()}

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
                      label=legend_scores_pca_qda, color='C6', marker='p')
    if 'lda' in scores:
        plot_with_var(axes,
                      mean=mean_scores['lda'], std=std_scores['lda'],
                      std_span=std_span,
                      label=legend_scores_lda, color='C3', marker='s')
    if 'pls_centroid' in scores:
        plot_with_var(axes,
                      mean=mean_scores['pls_centroid'],
                      std=std_scores['pls_centroid'],
                      std_span=std_span,
                      label=legend_scores_pls_centroid, color='C5', marker='X')
    if 'rkc' in scores:
        plot_with_var(axes,
                      mean=mean_scores['rkc'], std=std_scores['rkc'],
                      std_span=std_span,
                      label=legend_scores_rkc, color='C7', marker='*')
    axes.set_xticks(*list(zip(*[(i - start_pow, 2**i)
                                for i in range(start_pow, max_pow + 1)])))
    axes.set_xlabel("$N_b$")
    if plot_y_label:
        axes.set_ylabel("Accuracy")

    if plot_legend:
        leg = axes.legend(loc="upper left")
        leg.get_frame().set_alpha(1)

    axes.set_xlim(0, max_pow)
    axes.set_ylim(top=1.05)
    if ylim_bottom is not None:
        axes.set_ylim(bottom=ylim_bottom)
    axes.axhline(optimal_accuracy, linestyle=':', color='black')

    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmpfile:
        fig.savefig(tmpfile, format="pdf")
        if _run is not None:
            _run.add_artifact(tmpfile.name, name="plot.pdf")

    return fig


def plot_experiments_common(ids, function, titles=None, title=None):
    configure_matplotlib()

    n_experiments = len(ids)
    default_figsize = matplotlib.rcParams['figure.figsize']

    fig, axes = plt.subplots(1, n_experiments, figsize=(
        default_figsize[0] * n_experiments, default_figsize[1] * 1.5), sharey=True)

    for i, id in enumerate(ids):
        options = function(id)

        fig = plot_scores(**options,
                          _run=None,
                          plot_y_label=(i == 0),
                          plot_legend=False,
                          axes=axes[i])

        if titles is not None:
            axes[i].set_title(titles[i])

    if title is not None:
        fig.set_title(title)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center",
                     bbox_to_anchor=(0.5, 0),
                     bbox_transform=fig.transFigure, ncol=3)
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

    return result


def fdr_stat(confusion_matrix):
    print(confusion_matrix)

    return confusion_matrix[0, 1] / (
        confusion_matrix[0, 0] + confusion_matrix[0, 1])


def plot_confusion_matrix_stat(id, stat, plot_y_label=True):
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

    stat_dict = get_confusion_matrix_stat(confusion_matrices, stat)

    plot_scores(max_pow=max_pow,
                scores=stat_dict,
                legend_scores_optimal='Optimal',
                _run=None,
                optimal_accuracy=0,
                plot_y_label=plot_y_label)
