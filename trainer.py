from abc import ABC, abstractmethod
from .loggers.Printlogger import Printlogger
from tqdm import tqdm
from typing import Dict, Any
import torch
from torch import nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BaseTrainer(ABC):

    def __init__(self, model: nn.Module, loss_fn, optimizer, device=DEVICE, logger=Printlogger):
        super(BaseTrainer, self).__init__()
        self.device = DEVICE
        self.model = model.to(device)
        self.logger = logger
        self.loss_fn = loss_fn
        self.total_step = 0
        self.optimizer = optimizer

    def train_epoch(self, train_dataloader, epoch, pbar):
        self.model.train()
        losses = 0
        pbar.set_description(f'Training phase')
        for train_step, batch in enumerate(train_dataloader):
            loss = self.train_step(*batch if batch is tuple else batch, epoch=epoch, train_step=train_step)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses += loss.item()
            pbar.update()
            pbar.set_description(f'Training phase loss: {loss.item()}')
        return losses

    def evaluate(self, val_dataloader, epoch, pbar):
        pbar.set_description(f'Evaluating phase')
        self.model.eval()
        losses = 0
        with torch.no_grad():
            for val_step, batch in enumerate(val_dataloader):
                loss = self.val_step(*batch if batch is tuple else batch, epoch=epoch, train_step=val_step)
                losses += loss.item()
                pbar.update()
                pbar.set_description(f'Training phase loss: {loss.item()}')

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

    @abstractmethod
    def train_step(self, *args, **kwargs):
        raise NotImplementedError("create class for training don't use BaseTrainer")

    @abstractmethod
    def val_step(self, *args, **kwargs):
        raise NotImplementedError("create class for training don't use BaseTrainer")


