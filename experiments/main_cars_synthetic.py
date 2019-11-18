#!/usr/bin/env python3

from experiments.cars_synthetic.cars_synthetic import main, experiment


@experiment.automain
def automain():
    main()
