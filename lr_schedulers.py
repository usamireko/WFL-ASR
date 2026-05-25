import torch
from torch.optim.lr_scheduler import _LRScheduler
import pytorch_optimizer as optim


class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]


class WarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]
        return [base_lr for base_lr in self.base_lrs]


def get_scheduler(optimizer, scheduler_name, scheduler_params):
    if scheduler_name == "ConstantLR":
        return ConstantLR(optimizer)
    if scheduler_name == "WarmupLR":
        return WarmupLR(optimizer, **scheduler_params)

    try:
        scheduler_class = getattr(optim.lr_scheduler, scheduler_name)
        print(f"[INFO] Using scheduler '{scheduler_name}' from pytorch-optimizer.")
        return scheduler_class(optimizer, **scheduler_params)
    except AttributeError:
        try:
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name)
            print(f"[INFO] Using scheduler '{scheduler_name}' from torch.optim.lr_scheduler.")
            return scheduler_class(optimizer, **scheduler_params)
        except AttributeError as exc:
            raise ValueError(
                f"Scheduler '{scheduler_name}' not found in pytorch-optimizer or torch.optim.lr_scheduler"
            ) from exc
