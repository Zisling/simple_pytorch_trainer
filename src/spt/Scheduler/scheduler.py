import torch.optim.lr_scheduler
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import ReduceLROnPlateau


def get_normal_scheduler(scheduler_name, **kwargs):
    if scheduler_name == 'step':
        return lambda optimizer: StepLR(optimizer, **kwargs)
    elif scheduler_name == 'exp':
        return lambda optimizer: ExponentialLR(optimizer, **kwargs)
    elif scheduler_name == 'cos':
        return lambda optimizer: CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_name == 'coswarm':
        return lambda optimizer: CosineAnnealingWarmRestarts(optimizer, **kwargs)
    elif scheduler_name == 'plateau':
        return lambda optimizer: ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise NotImplementedError(f'Unknown scheduler: {scheduler_name}')
