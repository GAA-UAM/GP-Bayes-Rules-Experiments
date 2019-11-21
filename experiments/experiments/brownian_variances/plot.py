from ..common.plot import plot_experiments_common


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

    return {
        'max_pow': max_pow,
        'scores': scores,
        'optimal_accuracy': 1}


def plot_experiments(ids, **kwargs):
    return plot_experiments_common(ids, get_dict_by_id, **kwargs)
