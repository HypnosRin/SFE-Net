import torch.optim.lr_scheduler as lr_scheduler

from schedulers.cosine_annealing_lr_restart import CosineAnnealingLR_Restart


def get_scheduler(optim, opt):
    if opt['name'] is None:
        return None
    elif opt['name'] == 'ReduceLROnPlateau':
        return lr_scheduler.ReduceLROnPlateau(optim, factor=opt['factor'], patience=opt['patience'],
                                              min_lr=opt['min_lr'])
    elif opt['name'] == 'CosineAnnealingLR':
        return lr_scheduler.CosineAnnealingLR(optim, T_max=opt['T_max'], eta_min=opt['eta_min'])
    elif opt['name'] == 'CosineAnnealingWarmRestarts':
        return lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=opt['T_0'], T_mult=opt['T_mult'],
                                                        eta_min=opt['eta_min'])
    elif opt['name'] == 'CosineAnnealingLR_Restart':
        return CosineAnnealingLR_Restart(optim, opt)
    else:
        raise NotImplementedError
