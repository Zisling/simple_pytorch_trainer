from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn
from tqdm import tqdm

from .loggers.Loggers import PrintLogger

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BaseTrainer(ABC):

    def __init__(self, model: nn.Module, loss_fn, optimizer, device=DEVICE, logger=None):
        super(BaseTrainer, self).__init__()
        self.device = DEVICE
        self.model = model.to(device)
        self.logger = PrintLogger() if logger is None else logger
        self.loss_fn = loss_fn
        self.total_step = 0
        self.optimizer = optimizer

    def log(self, name, data, on_step=True):
        self.logger.log(name, data=data, on_step=on_step)

    def batch_to_device(self, batch) -> Tuple:
        if batch is not tuple:
            batch = (t.to(self.device) for t in batch)
        else:
            batch = (batch.to(self.device),)
        return batch

    def train_step_core(self, batch, **kwargs):
        batch = self.batch_to_device(batch)
        loss = self.train_step(*batch, **kwargs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def train_epoch(self, train_dataloader, epoch, pbar):
        self.model.train()
        losses = 0
        pbar.set_description(f'Training phase')
        for train_step, batch in enumerate(tqdm(train_dataloader, leave=False)):
            loss = self.train_step_core(batch, epoch=epoch, train_step=train_step)
            losses += loss.item()
            pbar.set_description(f'Training phase loss: {loss.item()}')
            pbar.update()
        # self.log('train/total_loss', losses / len(train_dataloader))
        return losses

    def evaluate(self, val_dataloader, epoch, pbar):
        pbar.set_description(f'Evaluating phase')
        self.model.eval()
        losses = 0
        with torch.no_grad():
            for val_step, batch in enumerate(tqdm(val_dataloader, leave=False)):
                batch = self.batch_to_device(batch)
                loss = self.val_step(*batch, epoch=epoch, val_step=val_step)
                losses += loss.item()
                pbar.set_description(f'Evaluating phase loss: {loss.item()}')
                pbar.update()
        # self.log('val/total_losses', losses / len(val_dataloader))
        return losses

    def fit(self, train_dataloader, val_dataloader, epoch_num=20):
        val_size = len(val_dataloader)
        train_size = len(train_dataloader)
        total_epoch_steps = train_size + val_size
        for epoch in tqdm(range(epoch_num)):
            with tqdm(total=total_epoch_steps, leave=False) as pbar:
                # training
                self.train_epoch(train_dataloader, epoch, pbar)
                # evaluating
                self.evaluate(val_dataloader, epoch, pbar)
            self.logger.epoc_end()
        self.logger.stop()

    @abstractmethod
    def train_step(self, *args, **kwargs):
        raise NotImplementedError("create class for training don't use BaseTrainer")

    @abstractmethod
    def val_step(self, *args, **kwargs):
        raise NotImplementedError("create class for training don't use BaseTrainer")
