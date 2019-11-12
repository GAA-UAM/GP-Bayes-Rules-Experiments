from ..common.plot import plot_scores


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

    scores = exp.info['scores']

    plot_scores(max_pow=max_pow,
                scores=scores,
                legend_scores_optimal='Step-Rule',
                _run=None,
                optimal_accuracy=1,
                plot_y_label=plot_y_label)
