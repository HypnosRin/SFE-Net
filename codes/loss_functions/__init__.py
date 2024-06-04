import torch.nn as nn


def get_loss_function(opt):
    if opt['name'] is None:
        return None
    elif opt['name'] == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif opt['name'] == 'L1':
        return nn.L1Loss()
    elif opt['name'] == 'MSE':
        return nn.MSELoss()
    else:
        raise NotImplementedError
