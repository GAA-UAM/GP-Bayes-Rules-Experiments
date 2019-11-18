import scipy.stats
import numpy as np

from ..common.plot import plot_scores


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


def plot_experiment(id, plot_y_label=True):
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

    plot_scores(max_pow=max_pow,
                scores=scores,
                legend_scores_optimal='Brownian-Bridge-Rule',
                _run=None,
                optimal_accuracy=1 - bayes_error(end_position),
                plot_y_label=plot_y_label)
