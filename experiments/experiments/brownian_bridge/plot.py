import tempfile

import scipy.stats

import matplotlib.pyplot as plt
import numpy as np

from ..common.classification import plot_with_var


def prob_error_brownian(end_position):
    dist = scipy.stats.norm(scale=np.sqrt(end_position))

    D = np.sqrt(-np.log(1 - end_position) * (1 - end_position))

    return dist.cdf(D) - dist.cdf(-D)


def prob_error_brownian_bridge(end_position):
    dist = scipy.stats.norm(scale=np.sqrt(end_position - end_position**2))

    D = np.sqrt(-np.log(1 - end_position) * (1 - end_position))

    return 2 * dist.cdf(-D)


def bayes_error(end_position):
    return (0.5 * prob_error_brownian(end_position)
            + 0.5 * prob_error_brownian_bridge(end_position))


def plot_scores(max_pow, end_position, scores, scores_lda, scores_qda,
                scores_pls_centroid, scores_galeano, scores_rkc, _run):
    mean_scores = np.mean(scores, axis=1)
    mean_scores_lda = np.mean(scores_lda, axis=1)
    mean_scores_qda = np.mean(scores_qda, axis=1)
    mean_scores_pls_centroid = np.mean(scores_pls_centroid, axis=1)
    mean_scores_galeano = np.mean(scores_galeano, axis=1)
    mean_scores_rkc = np.mean(scores_rkc, axis=1)

    std_scores = np.std(scores, axis=1)
    std_scores_lda = np.std(scores_lda, axis=1)
    std_scores_qda = np.std(scores_qda, axis=1)
    std_scores_pls_centroid = np.std(scores_pls_centroid, axis=1)
    std_scores_galeano = np.std(scores_galeano, axis=1)
    std_scores_rkc = np.std(scores_rkc, axis=1)

    legend_scores = 'Brownian-Bridge rule'
    legend_scores_lda = 'LDA'
    legend_scores_qda = 'QDA'
    legend_scores_pls_centroid = 'PLS+Centroid'
    legend_scores_galeano = 'PCA+QDA'
    legend_scores_rkc = 'RKC'

    # plt.title('Accuracy')
    plt.figure()

    std_span = 1

    plot_with_var(mean=mean_scores, std=std_scores,
                  std_span=std_span,
                  label=legend_scores, color='C0', linestyle=':', marker='o')
    plot_with_var(mean=mean_scores_qda, std=std_scores_qda,
                  label=legend_scores_qda,
                  std_span=std_span,
                  color='C2', linestyle='-.', marker='v')
    plot_with_var(mean=mean_scores_lda, std=std_scores_lda,
                  std_span=std_span,
                  label=legend_scores_lda, color='C3', marker='s')
    plot_with_var(mean=mean_scores_pls_centroid, std=std_scores_pls_centroid,
                  std_span=std_span,
                  label=legend_scores_pls_centroid, color='C5', marker='X')
    plot_with_var(mean=mean_scores_galeano, std=std_scores_galeano,
                  std_span=std_span,
                  label=legend_scores_galeano, color='C6', marker='p')
    plot_with_var(mean=mean_scores_rkc, std=std_scores_rkc,
                  std_span=std_span,
                  label=legend_scores_rkc, color='C7', marker='*')
    plt.xticks(*list(zip(*[(i - 1, 2**i)
                           for i in range(1, max_pow + 1)])))
    plt.xlabel("$N_b$")
    plt.ylabel("Accuracy")
    leg = plt.legend()
    leg.get_frame().set_alpha(1)

    plt.xlim(0, max_pow)
    plt.ylim(top=1.05)
    plt.axhline(1 - bayes_error(end_position), linestyle=':', color='black')

    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmpfile:
        plt.savefig(tmpfile, format="pdf")
        if _run is not None:
            _run.add_artifact(tmpfile.name, name="plot.pdf")


def plot_experiment(id):
    from incense import ExperimentLoader

    loader = ExperimentLoader(
        # None if MongoDB is running on localhost or "mongodb://mongo:27017"
        # when running in devcontainer.
        mongo_uri=None,
        db_name='GPBayes'
    )

    exp = loader.find_by_id(id)

    max_pow = exp.config['max_pow']
    end_position = exp.config['end_position']

    scores = exp.info['scores']
    scores_lda = exp.info['scores_lda']
    scores_qda = exp.info['scores_qda']
    scores_pls_centroid = exp.info['scores_pls_centroid']
    scores_galeano = exp.info['scores_galeano']
    scores_rkc = exp.info['scores_rkc']

    plot_scores(max_pow=max_pow, end_position=end_position,
                scores=scores,
                scores_lda=scores_lda,
                scores_qda=scores_qda,
                scores_pls_centroid=scores_pls_centroid,
                scores_galeano=scores_galeano,
                scores_rkc=scores_rkc,
                _run=None)
