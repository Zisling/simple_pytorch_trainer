from abc import ABC, abstractmethod

import torch
from torch import nn
from tqdm import tqdm
from typing import Dict, Any
import neptune.new as neptune
import matplotlib.pyplot as plt


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class LoggerHolder(metaclass=Singleton):

    def __init__(self, logger):
        super(LoggerHolder, self).__init__()
        self.logger = logger if logger is not None else None

    def get_logger(self):
        return self.logger


class PrintLogger(ABC):

    def __init__(self, metrics_to_track=None, **kwargs):
        super(PrintLogger, self).__init__()
        if metrics_to_track is None:
            metrics_to_track = []
        self.metrics_to_track = metrics_to_track
        self.name_value = {}  # name of log to the list of values
        self.name_step = {}  # name of log to the number of logs call on that parameter
        self.tracked_metrics = {}  # metric to save on epoc
        self.state = {}

    def set_state(self, state):
        self.state = state

    def set_metrics_to_track(self, metrics_to_track):
        self.metrics_to_track = metrics_to_track

    def __enter__(self):
        return self

    def stop(self):
        pass

    def __exit__(self, type, value, traceback):
        self.stop()

    def log(self, name: str, data: Any, on_step=True, average_over_epoch=False):
        if on_step:
            self._log(name, data)
        if average_over_epoch:
            self._epoch_log(name, data)
        if name in self.metrics_to_track:
            self.tracked_metrics[name] = data

    def _log(self, name: str, data: Any):
        tqdm.write(f'{name}: {data}')

    def log_fig(self, name: str, fig: Any):
        plt.show()

    def _epoch_log(self, name: str, data: Any):
        if name in self.name_value:
            self.name_value[name] += data
            self.name_step[name] += 1
        else:
            self.name_value[name] = data
            self.name_step[name] = 1

    def epoch_end(self):
        """
        This will log all values with that where log with the flag average_over_epoch
        and if the metrics is tracked will also save this to tracked_metrics dict
        :return:
        """
        for name, value in self.name_value.items():
            mean = value / self.name_step[name]
            name_split = name.split('/')
            name_split[-1] = 'mean_' + name_split[-1]
            self._log('/'.join(name_split), mean)
            if name in self.metrics_to_track:
                self.tracked_metrics[name] = mean
        self.name_value = {}
        self.name_step = {}

    def get_tracked_metrics(self) -> Dict:
        return self.tracked_metrics

    def upload_model(self, path):
        pass

    def save_param(self, name: str, data: Any):
        self.tracked_metrics[name] = data
        self._save_param(name, data)

    def _save_param(self, name: str, data: Any):
        tqdm.write(f'{name}: {data}')


class NeptuneLogger(PrintLogger):

    def __init__(self, project_name, api_token, params=None, **kwargs):
        super(NeptuneLogger, self).__init__(**kwargs)
        self.run = neptune.init(project=project_name, api_token=api_token)
        self.run["parameters"] = params

    def _log(self, name: str, data: Any):
        self.run[name].log(data)

    def log_fig(self, name: str, fig: Any):
        self._log(name, fig)

    def stop(self):
        self.run.stop()

    def upload_model(self, path):
        self.run['model_weights'].upload(path)

    def _save_param(self, name: str, data: Any):
        self.run[name] = data


def Loggable(original_class):
    orig_init = original_class.__init__

    # Make copy of original __init__, so we can call it without recursion

    def __init__(self, *args, **kws):
        orig_init(self, *args, **kws)  # Call the original __init__

    def log(self, *args, **kwargs):
        logger = LoggerHolder(None).get_logger()
        logger.log(*args, **kwargs)

    def get_logger(self):
        return LoggerHolder(None).get_logger()

    original_class.__init__ = __init__  # Set the class' __init__ to the new one
    original_class.log = log  # Set the class' log to the new one
    original_class.get_logger = get_logger  # Set the class' log to the new one
    return original_class
