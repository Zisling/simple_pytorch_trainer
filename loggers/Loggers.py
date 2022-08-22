from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from typing import Dict, Any
import neptune.new as neptune


class PrintLogger(ABC):

    def __init__(self):
        super(PrintLogger, self).__init__()
        self.name_value = {}

    def __enter__(self):
        return self

    def log(self, name: str, data: Any, on_step=True):
        if on_step:
            self._log(name, data)
        else:
            self._epoch_log(name, data)

    def _log(self, name: str, data: Any):
        tqdm.write(f'{name}: {data}')

    def _epoch_log(self, name: str, data: Any):
        if name in self.name_value:
            self.name_value[name] += data
        else:
            self.name_value[name] = data

    def epoc_end(self):
        for name, value in self.name_value.items():
            self._log(name, np.mean(value))
        self.name_value = {}

    def stop(self):
        pass

    def __exit__(self):
        self.stop()

    def save_model(self, model: nn.Module, path: str):
        torch.save(model.state_dict(), path)


class NeptuneLogger(PrintLogger):

    def __init__(self, project_name, api_token, params=None):
        super(NeptuneLogger, self).__init__()
        self.run = neptune.init(project=project_name, api_token=api_token)
        self.run["parameters"] = params

    def _log(self, name: str, data: Any):
        self.run[name].log(data)

    def stop(self):
        self.run.stop()
