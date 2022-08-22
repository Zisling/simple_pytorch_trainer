import torch
from torch import nn
from abc import ABC, abstractmethod
from torch.optim.optimizer import Optimizer
from typing import Dict, List
import os
from pathlib import Path


def get_full_path(path: str, name: str, suffix: str):
    return str(Path(os.path.join(path, name)).with_suffix(suffix))


class BaseCheckPoint(ABC):

    def __init__(self, path: str, metrics_to_track: List):
        self.metrics_to_track = metrics_to_track
        os.makedirs(path, exist_ok=True)
        self.path = path

    @abstractmethod
    def save_callback(self, metrics: Dict, epoch: int, model: nn.Module, optimizer: Optimizer,
                      *args, **kwargs) -> str:
        pass

    @staticmethod
    def save_model(model: nn.Module, path: str, name: str) -> str:
        """
        this function is for saving the model state_dict
        .pth file extension
        https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
        :param model: pytorch model
        :param path: the dir path
        :param name: the name to save with
        :return: the full path of where the model is saved
        """
        full_path = get_full_path(path, name, '.pth')
        torch.save(model.state_dict(), full_path)
        return full_path

    @staticmethod
    def save_entire_model(model: nn.Module, path: str, name: str) -> str:
        """
        this is the full model and to load you need the model class imported and load the model.
        model = torch.load(path)
        we don't use this file in load_model function just state dict
        https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model
        :param model: pytorch model
        :param path: the full path including the file name and the file extension
        :param name:
        :return: the full path of where the model is saved
        """
        full_path = get_full_path(path, name, '.pth')
        torch.save(model, full_path)
        return full_path
    @staticmethod
    def save_torch_script(model: nn.Module, path: str, name: str) -> str:
        """
        This save the model in torch script
        .pt file extension
        https://pytorch.org/tutorials/beginner/saving_loading_models.html#export-load-model-in-torchscript-format
        :param model: pytorch model
        :param path: the full path including the file name and the file extension
        :param name:
        :return: the full path of where the model is saved
        """
        full_path = get_full_path(path, name, '.pth')
        model_scripted = torch.jit.script(model)
        model_scripted.save(full_path)
        return full_path

    @staticmethod
    def save_check_point(model: nn.Module, optimizer: Optimizer, epoch, path, name, **kwargs) -> str:
        """
        https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
        :param model: pytorch model
        :param optimizer: pytorch optimizer
        :param epoch: the epoch number
        :param path: the full path including the file name and the file extension
        :param name:
        :return: the full path of where the model is saved
        """
        full_path = get_full_path(path, name, '.pth')
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    **kwargs},
                   full_path)
        return full_path
