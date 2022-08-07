from abc import ABC, abstractmethod
from tqdm import tqdm
from typing import Dict, Any
import neptune.new as neptune


class PrintLogger(ABC):

    def __init__(self):
        super(PrintLogger, self).__init__()

    def __enter__(self):
        return self

    def log(self, name: str, data: Any):
        tqdm.write(f'{name}: {data}')

    def stop(self):
        pass

    def __exit__(self):
        self.stop()


class NeptuneLogger(PrintLogger):

    def __init__(self, project_name, api_token, Parmas=None):
        super(NeptuneLogger, self).__init__()
        self.run = neptune.init(project=project_name, api_token=api_token)
        self.run["parameters"] = Parmas

    def log(self, name: str, data: Any):
        self.run[name].log(data)

    def stop(self):
        self.run.stop()
