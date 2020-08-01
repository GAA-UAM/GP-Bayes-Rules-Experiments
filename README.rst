GP-Bayes-Rules-Experiments
===================================================

Experiments in the empirical evaluation section of the article [OPT2020]_.

.. [OPT2020] Torrecilla, J. L., Ramos-Carreño, C., Sánchez-Montañés, M., 
   & Suárez, A. (2020). Optimal classification of Gaussian processes in homo-
   and heteroscedastic settings. Statistics and Computing, 30(4), 1091-1111.
   https://doi.org/10.1007/s11222-020-09937-7

Installation
============

These experiments require that the library *scikit-fda* and its dependencies
are installed.
The stable version can be installed via PyPI:

.. code::

    pip install scikit-fda
    
The experiments require also the package *Sacred* to collect the results.
This package is also available in PyPI:

.. code::

    pip install sacred

Sacred requires an observer to store the results. The functions used to query
the results and plot the output assume that the observer used is a MongoDB
database. In that case, MongoDB should be
`installed <https://docs.mongodb.com/manual/administration/install-community/>`_
and the packages *incense* and *pymongo* should also be installed to retrieve
the results:

.. code::

    pip install incense
    pip install pymongo
    
Launching a experiment
======================

In order to launch an experiment, we will execute the main files in the
project folder. As these experiments use Sacred to run, the configuration
options can be set to match the ones used in the article (or to test a
different configuration). An example of this is shown below:

.. code::

	python main_brownian_bridge.py with train_n_samples=50 -m localhost:27017:GPBayes
	
The ``with`` keyword is used to change the options of the experiment. The
``-m localhost:27017:GPBayes`` adds a MongoDB observer, storing the results in
the *GPBayes* database. It is important to use this name as it is currently
harcoded in the retrieval functions.

Plotting a experiment
=====================

In order to plot the results, there are functions called `plot_experiments` in
each of the submodules corresponding to the plotting part to each experiment.
These functions can create the *matplotlib* figure of the results, as shown
below:

.. code:: python

	from experiments.brownian_bridge.plot import plot_experiments
	import matplotlib.pyplot as plt
	
	plot_experiments([1, 2, 3])
	plt.show()
	
Here we assume that the sacred experiments with ids 1, 2 and 3 contain,
respectively, the results of the Brownian bridge experiment with train sizes
50, 200 and 1000, in order to replicate the results of the paper.
	
List of experiments
===================

The list of the experiments with synthetic data and their configuration 
parameters is shown below. A comprehensive description of each experiment
is in the original article.

The common configuration parameters are the following:

- ``max_pow = 10``: The maximum power of the resolution used in the
	discretization.
- ``n_tests = 100``: The number of independent replications.
- ``train_n_samples = 1000``: The number of observations in the train set. It
	must be set to 50, 200 and 1000 in separate runs, to replicate the
	results of the paper.
- ``test_n_samples = 1000``: The number of observations in the test set. 
- ``random_state_train_seed = 0``: A random seed to initialize the RNG that
	is used in the train set generation.
- ``random_state_test_seed = 1``: A random seed to initialize the RNG that
	is used in the test set generation.

Brownian processes with different means
---------------------------------------

This experiment presents a classification problem in which the classes are two
Brownian processes with different means. One of the means is 0 and the other
is a step function. The experiment folder for this experiment is
``brownian_step``.

The additional configuration parameters for this experiment are:

- ``step_height = 0.3``: The height of the function after the step.

Brownian motion versus Brownian bridge
--------------------------------------

This experiment presents a classification problem in which the classes are a
standard Brownian process and a standard Brownian bridge process. The
experiment folder for this experiment is ``brownian_bridge``.

The additional configuration parameters for this experiment are:

- ``end_position = 0.5``: The end of the interval in which the functions are
	evaluated. Set it to 0.95 to match the results in the article.
	
Brownian processes with different variances
-------------------------------------------

This experiment presents a classification problem in which the classes are two
Brownian processes with different variances. The experiment folder for this
experiment is ``brownian_variances``.

The additional configuration parameters for this experiment are:

- ``class0_var = 1``: The variance of class 0.
- ``class1_var = 1.3``: The variance of class 1. Set it to 1.5 to match the
  results in the article.
  
The real data example and the simulated data example, available in ``cars``
and ``cars_synthetic`` are similar to this one. The data for the real
data example cannot be publicly posted, as it came from Google Finance.
Contact the maintainer for more info.