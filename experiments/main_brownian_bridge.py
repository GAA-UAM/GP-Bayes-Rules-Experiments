#!/usr/bin/env python3

from experiments.brownian_bridge.brownian_bridge import main, experiment


@experiment.automain
def automain():
    main()
