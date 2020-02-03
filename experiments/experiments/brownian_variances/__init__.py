import os

from sacred import Experiment
from sacred.observers import TelegramObserver


experiment = Experiment(name="brownian_variances")

# FILE_PATH = os.path.dirname(__file__)
# 
# telegram_obs = TelegramObserver.from_config(os.path.join(FILE_PATH, '..',
#                                                          'telegram.json'))
# experiment.observers.append(telegram_obs)
