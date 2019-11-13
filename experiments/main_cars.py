#!/usr/bin/env python3

from experiments.cars.cars import main, experiment


@experiment.automain
def automain():
    main()
