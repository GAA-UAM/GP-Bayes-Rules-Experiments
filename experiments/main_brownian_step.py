#!/usr/bin/env python3

from experiments.brownian_step.brownian_step import main, experiment


@experiment.automain
def automain():
    main()
