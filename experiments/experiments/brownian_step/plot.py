import tempfile

import matplotlib.pyplot as plt
import numpy as np

from ..common.classification import plot_with_var


def plot_scores(max_pow, scores, legend_scores_optimal, _run):

    mean_scores = {key: np.mean(s, axis=1) for key, s in scores.items()}
    std_scores = {key: np.std(s, axis=1) for key, s in scores.items()}

    legend_scores_lda = 'LDA'
    legend_scores_qda = 'QDA'
    legend_scores_pls_centroid = 'PLS+Centroid'
    legend_scores_pca_qda = 'PCA+QDA'
    legend_scores_rkc = 'RKC'

    # plt.title('Accuracy')
    plt.figure()

    std_span = 1

    plot_with_var(mean=mean_scores['optimal'], std=std_scores['optimal'],
                  std_span=std_span,
                  label=legend_scores_optimal, color='C0', linestyle=':',
                  marker='o')
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
    plt.xticks(*list(zip(*[(i - 1, 2**i)
                           for i in range(1, max_pow + 1)])))
    plt.xlabel("$N_b$")
    plt.ylabel("Accuracy")
    leg = plt.legend(loc="upper left")
    leg.get_frame().set_alpha(1)

    plt.xlim(0, max_pow)
    plt.ylim(top=1.05)
    plt.axhline(1, linestyle=':', color='black')

    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmpfile:
        plt.savefig(tmpfile, format="pdf")
        if _run is not None:
            _run.add_artifact(tmpfile.name, name="plot.pdf")

    plt.show()


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

    scores = exp.info['scores']
    scores_lda = exp.info['scores_lda']
    scores_qda = exp.info['scores_qda']
    scores_pls_centroid = exp.info['scores_pls_centroid']
    scores_galeano = exp.info['scores_galeano']
    scores_rkc = exp.info['scores_rkc']

    plot_scores(max_pow=max_pow,
                scores=scores,
                scores_lda=scores_lda,
                scores_qda=scores_qda,
                scores_pls_centroid=scores_pls_centroid,
                scores_galeano=scores_galeano,
                scores_rkc=scores_rkc,
                _run=None)
