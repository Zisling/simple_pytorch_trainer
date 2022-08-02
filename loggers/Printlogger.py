from abc import ABC, abstractmethod
from tqdm import tqdm
from typing import Dict, Any


class Printlogger(ABC):

    def __init__(self):
        super(Printlogger, self).__init__()

    def dict_log(self, name_data: Dict[str, Any]):
        for name, data in name_data.items():
            self.log(name, data)

    def log(self, name: str, data: Any):
        tqdm.write(f'{name}: {data}')
