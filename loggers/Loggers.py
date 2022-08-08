from abc import ABC, abstractmethod
from tqdm import tqdm
from typing import Dict, Any
import neptune.new as neptune


class PrintLogger(ABC):

    def __init__(self):
        super(PrintLogger, self).__init__()
        self.name_value = {}
        self.name_step = {}

    def __enter__(self):
        return self

    def log(self, name: str, data: Any, on_step=True):
        if on_step:
            self._log(name, data)
        else:
            self._epoc_log(name, data)

    def _log(self, name: str, data: Any):
        tqdm.write(f'{name}: {data}')

    def _epoc_log(self, name: str, data: Any):
        if name in self.name_value:
            self.name_value[name] += data
            self.name_step[name] += 1
        else:
            self.name_value[name] = data
            self.name_step[name] = 1

    def epoc_end(self):
        for name, value in self.name_value.items():
            self._log(name, value / self.name_step[name])
        self.name_value = {}
        self.name_step = {}

    def stop(self):
        pass

    def __exit__(self):
        self.stop()


class NeptuneLogger(PrintLogger):

    def __init__(self, project_name, api_token, Parmas=None):
        super(NeptuneLogger, self).__init__()
        self.run = neptune.init(project=project_name, api_token=api_token)
        self.run["parameters"] = Parmas

    def _log(self, name: str, data: Any):
        self.run[name].log(data)

    def stop(self):
        self.run.stop()
