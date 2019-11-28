#!/usr/bin/env python3

from experiments.brownian_variances_simulation.brownian_variances_simulation import main, experiment


@experiment.automain
def automain():
    main()
