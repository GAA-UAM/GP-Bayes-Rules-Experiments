import scipy.stats
import numpy as np

from ..common.plot import plot_experiments_common


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
    end_position = exp.config['end_position']

    scores = exp.info['scores']

    return {
        'max_pow': max_pow,
        'scores': scores,
        'optimal_accuracy': 1 - bayes_error(end_position)}


def plot_experiments(ids, **kwargs):
    return plot_experiments_common(ids, get_dict_by_id, **kwargs)
