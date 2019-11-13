#!/usr/bin/env python3

from experiments.brownian_variances.brownian_variances import main, experiment


@experiment.automain
def automain():
    main()
