import torch.optim as optim


def get_optimizer(params, opt):
    if opt['name'] is None:
        return None
    elif opt['name'] == 'SGD':
        return optim.SGD(params=params, lr=opt['lr'], weight_decay=opt['weight_decay'])
    elif opt['name'] == 'Adam':
        return optim.Adam(params=params, lr=opt['lr'], weight_decay=opt['weight_decay'])
    else:
        raise NotImplementedError
