"""
scheduler.py - Noam Learning Rate Scheduler
DA6401 Assignment 3

lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler


class NoamScheduler(_LRScheduler):
    """
    Noam learning rate schedule from "Attention Is All You Need" Section 5.3.

    lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))

    Usage:
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, d_model=256, warmup_steps=4000)
        # call scheduler.step() after every optimizer.step()
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step_num = 0  # internal counter (1-indexed in the formula)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Return the learning rate for the current step."""
        step = max(1, self._step_num)
        scale = (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5)
        )
        return [scale for _ in self.base_lrs]

    def step(self, epoch=None):
        self._step_num += 1
        super().step(epoch)

    @property
    def current_lr(self):
        return self.get_lr()[0]