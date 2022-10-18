from abc import ABC, abstractmethod
from typing import Tuple
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from trainer.src.spt.loggers.Loggers import PrintLogger

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BaseTrainer(ABC):

    def __init__(self, model: nn.Module, loss_fn, optimizer: Optimizer, device=DEVICE, logger=None, checkpoint=None,
                 disable_progress_bar=False, fp16=False):
        super(BaseTrainer, self).__init__()
        self.device = DEVICE
        self.model = model.to(device)
        if fp16:
            self.model = model.half()
        self.logger = PrintLogger() if logger is None else logger
        self.loss_fn = loss_fn
        self.total_step = 0
        self.optimizer = optimizer
        if checkpoint is not None:
            self.logger.set_metrics_to_track(checkpoint.metrics_to_track)
        self.checkpoint = checkpoint
        self.disable_pbar = disable_progress_bar
        self.fp16 = fp16

    def log(self, name, data, on_step=True, average_over_epoch=False):
        self.logger.log(name, data=data, on_step=on_step, average_over_epoch=average_over_epoch)

    def batch_to_device(self, batch) -> Tuple:
        if batch is not tuple:
            if self.fp16:
                batch = (t.to(self.device).half() if (t.dtype is torch.FloatTensor or t.dtype is torch.DoubleTensor)
                         else t.to(self.device)
                         for t in batch)
            else:
                batch = (t.to(self.device) for t in batch)
        else:
            if self.fp16:
                batch = (batch.to(self.device).half(),) \
                    if (batch.dtype is torch.FloatTensor or batch.dtype is torch.DoubleTensor) \
                    else (batch.to(self.device), )
            else:
                batch = (batch.to(self.device),)
        return batch

    def train_step_core(self, batch, **kwargs):
        batch = self.batch_to_device(batch)
        self.optimizer.zero_grad()
        loss = self.train_step(*batch, **kwargs)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_epoch(self, train_dataloader, epoch, pbar):
        self.model.train()
        losses = 0
        pbar.set_description(f'Training phase')
        for train_step, batch in enumerate(tqdm(train_dataloader, leave=False, disable=self.disable_pbar)):
            loss = self.train_step_core(batch, epoch=epoch, train_step=train_step)
            losses += loss
            pbar.set_description(f'Training phase loss: {loss}')
            pbar.update()
        return losses

    def evaluate(self, val_dataloader, epoch, pbar):
        pbar.set_description(f'Evaluating phase')
        self.model.eval()
        losses = 0
        with torch.no_grad():
            for val_step, batch in enumerate(tqdm(val_dataloader, leave=False, disable=self.disable_pbar)):
                batch = self.batch_to_device(batch)
                loss = self.val_step(*batch, epoch=epoch, val_step=val_step)
                losses += loss.item()
                pbar.set_description(f'Evaluating phase loss: {loss.item()}')
                pbar.update()
        return losses

    def fit(self, train_dataloader, val_dataloader, epoch_num=20):
        val_size = len(val_dataloader)
        train_size = len(train_dataloader)
        total_epoch_steps = train_size + val_size
        save_path = None
        for epoch in tqdm(range(epoch_num), disable=self.disable_pbar):
            with tqdm(total=total_epoch_steps, leave=False, disable=self.disable_pbar) as pbar:
                # training
                self.train_epoch(train_dataloader, epoch, pbar)
                # evaluating
                self.evaluate(val_dataloader, epoch, pbar)
                self.logger.epoch_end()
                self.val_epoch_end(epoch)
                save_path = self.save_callback(epoch)
        if save_path is not None:
            self.logger.upload_model(save_path)

    @abstractmethod
    def train_step(self, *args, **kwargs):
        raise NotImplementedError("create class for training don't use BaseTrainer")

    @abstractmethod
    def val_step(self, *args, **kwargs):
        raise NotImplementedError("create class for training don't use BaseTrainer")

    def val_epoch_end(self, epoch: int):
        pass

    # Model saving

    def save_model(self, path: str):
        """
        this function is for saving the model state_dict
        .pth file extension
        https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
        :param path: the full path including the file name and the file extension
        :return:
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))

    def save_entire_model(self, path: str):
        """
        this is the full model and to load you need the model class imported and load the model.
        model = torch.load(path)
        we don't use this file in load_model function just state dict
        https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model
        :param path: the full path including the file name and the file extension
        :return:
        """
        torch.save(self.model, path)

    def save_torch_script(self, path: str):
        """
        This save the model in torch script
        .pt file extension
        https://pytorch.org/tutorials/beginner/saving_loading_models.html#export-load-model-in-torchscript-format
        :param path: the full path including the file name and the file extension
        :return:
        """
        model_scripted = torch.jit.script(self.model)
        model_scripted.save(path)

    def load_torch_script(self, path: str):
        """
        this load the model in torch script and overwrite the current model
        :param path: the full path including the file name and the file extension
        :return:
        """
        self.model = torch.jit.load(path)

    def save_check_point(self, epoch, path, **kwargs):
        """
        https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
        :param epoch: the epoch number
        :param path: the full path including the file name and the file extension
        :return:
        """
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    **kwargs},
                   path)

    def save_callback(self, epoch):
        if self.checkpoint is not None:
            return self.checkpoint.save_callback(self.logger.get_tracked_metrics(), epoch, self.model, self.optimizer)
        return None
