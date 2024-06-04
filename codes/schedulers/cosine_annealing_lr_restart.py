import math
import torch
from torch.optim.lr_scheduler import _LRScheduler  # 这行IDE可能报错，但实际运行时没错


class CosineAnnealingLR_Restart(_LRScheduler):
    def __init__(self, optimizer, opt: dict):
        self.T_period = opt['T_period']
        self.restarts = opt['restarts'] if opt['restarts'] else [0]
        self.weights = opt['weights'] if opt['weights'] else [1]
        self.eta_min = opt['eta_min']
        self.last_epoch = -1
        self.T_max = self.T_period[0]  # current T period
        self.last_restart = 0
        assert len(self.restarts) == len(self.weights), 'restarts and their weights do not match.'
        super().__init__(optimizer, self.last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch in self.restarts:
            self.last_restart = self.last_epoch
            self.T_max = self.T_period[self.restarts.index(self.last_epoch) + 1]
            weight = self.weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        elif (self.last_epoch - self.last_restart - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_max)) /
                (1 + math.cos(math.pi * ((self.last_epoch - self.last_restart) - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min for group in self.optimizer.param_groups]


def main():
    optimizer = torch.optim.Adam(params=[torch.rand((3, 3))])
    scheduler = CosineAnnealingLR_Restart(optimizer, opt={'T_period': [12500, 12500, 12500, 12500],
                                                          'restarts': [12500, 25000, 37500],
                                                          'weights': [1, 1, 1],
                                                          'eta_min': 1e-7})
    print(scheduler)


if __name__ == '__main__':
    main()
