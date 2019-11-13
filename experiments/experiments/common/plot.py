import tempfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def configure_matplotlib():
    plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['text.usetex'] = True


def plot_with_var(mean, std, color, label, std_span=0, **kwargs):

    for multiple in range(std_span, 0, -1):
        plt.fill_between(range(len(mean)), mean - multiple *
                         std, mean + multiple * std, color=color, alpha=0.15)
    plt.plot(mean, label=label, color=color, **kwargs)


def plot_scores(max_pow, scores, legend_scores_optimal, _run, optimal_accuracy,
                std_span=1, plot_y_label=True,
                theoretical_mean=None, theoretical_std=None,
                start_pow=1):

    configure_matplotlib()

    mean_scores = {key: np.mean(s, axis=1) for key, s in scores.items()}
    std_scores = {key: np.std(s, axis=1) for key, s in scores.items()}

    legend_scores_brownian_qda = 'Brownian-QDA'
    legend_scores_lda = 'LDA'
    legend_scores_qda = 'QDA'
    legend_scores_pls_centroid = 'PLS+Centroid'
    legend_scores_pca_qda = 'PCA+QDA'
    legend_scores_rkc = 'RKC'
    legend_theoretical = 'Theoretical'

    # plt.title('Accuracy')
    plt.figure()

    if theoretical_mean is not None:
        plot_with_var(mean=theoretical_mean,
                      std=theoretical_std,
                      std_span=2,
                      label=legend_theoretical, color='C4')
    plot_with_var(mean=mean_scores['optimal'], std=std_scores['optimal'],
                  std_span=std_span,
                  label=legend_scores_optimal, color='C0', linestyle=':',
                  marker='o')
    if 'brownian_qda' in scores:
        plot_with_var(mean=mean_scores['brownian_qda'],
                      std=std_scores['brownian_qda'],
                      label=legend_scores_brownian_qda,
                      std_span=std_span,
                      color='C1', linestyle='--', marker='^')
    plot_with_var(mean=mean_scores['qda'], std=std_scores['qda'],
                  label=legend_scores_qda,
                  std_span=std_span,
                  color='C2', linestyle='-.', marker='v')
    plot_with_var(mean=mean_scores['pca_qda'], std=std_scores['pca_qda'],
                  std_span=std_span,
                  label=legend_scores_pca_qda, color='C6', marker='p')
    plot_with_var(mean=mean_scores['lda'], std=std_scores['lda'],
                  std_span=std_span,
                  label=legend_scores_lda, color='C3', marker='s')
    plot_with_var(mean=mean_scores['pls_centroid'],
                  std=std_scores['pls_centroid'],
                  std_span=std_span,
                  label=legend_scores_pls_centroid, color='C5', marker='X')
    plot_with_var(mean=mean_scores['rkc'], std=std_scores['rkc'],
                  std_span=std_span,
                  label=legend_scores_rkc, color='C7', marker='*')
    plt.xticks(*list(zip(*[(i - start_pow, 2**i)
                           for i in range(start_pow, max_pow + 1)])))
    plt.xlabel("$N_b$")
    if plot_y_label:
        plt.ylabel("Accuracy")
    leg = plt.legend(loc="upper left")
    leg.get_frame().set_alpha(1)

    plt.xlim(0, max_pow)
    plt.ylim(top=1.05)
    plt.axhline(optimal_accuracy, linestyle=':', color='black')

    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmpfile:
        plt.savefig(tmpfile, format="pdf")
        if _run is not None:
            _run.add_artifact(tmpfile.name, name="plot.pdf")

    plt.show()
