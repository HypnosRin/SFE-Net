import torch.nn as nn
from torch.nn import init
import functools

from networks.IKC import SFTMD, Predictor, Corrector
from networks.MANet import MANet_s1
from networks.unet_based import FFTRCANResUNet
from networks.unet_sr import UNetSR2
from networks.DFCAN import SFTDFCAN, DFCAN, SFTDFCAN_sequential


def init_weights(net, init_conv_linear_type='orthogonal', gain=0.2, init_bn_type='uniform'):
    """init weights of Conv, Linear and BatchNorm"""

    def init_fn(layer, initial_type, my_gain, initial_bn_type):
        if type(layer) in (nn.Conv2d, nn.ConvTranspose2d, nn.Linear):
            if initial_type == 'normal':
                init.normal_(layer.weight.data, 0, 0.1)
                layer.weight.data.clamp_(-1, 1).mul_(my_gain)
            elif initial_type == 'uniform':
                init.uniform_(layer.weight.data, -0.2, 0.2)
                layer.weight.data.mul_(my_gain)
            elif initial_type == 'xavier_normal':
                init.xavier_normal_(layer.weight.data, gain=my_gain)
                layer.weight.data.clamp_(-1, 1)
            elif initial_type == 'xavier_uniform':
                init.xavier_uniform_(layer.weight.data, gain=my_gain)
            elif initial_type == 'kaiming_normal':
                init.kaiming_normal_(layer.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                layer.weight.data.clamp_(-1, 1).mul_(my_gain)
            elif initial_type == 'kaiming_uniform':
                init.kaiming_uniform_(layer.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                layer.weight.data.mul_(my_gain)
            elif initial_type == 'orthogonal':
                init.orthogonal_(layer.weight.data, gain=my_gain)
            else:
                raise RuntimeWarning(f'initialization method [{initial_type}] is not implemented')
            if layer.bias is not None:
                layer.bias.data.zero_()
        elif type(layer) in (nn.BatchNorm2d, nn.InstanceNorm2d):
            if initial_bn_type == 'uniform':  # preferred
                if layer.affine:
                    init.uniform_(layer.weight.data, 0.1, 1.0)
                    init.constant_(layer.bias.data, 0.0)
            elif initial_bn_type == 'constant':
                if layer.affine:
                    init.constant_(layer.weight.data, 1.0)
                    init.constant_(layer.bias.data, 0.0)
            elif initial_bn_type == 'normal':
                if layer.affine:
                    init.normal_(layer.weight.data, 1.0, 0.02)
                    init.constant_(layer.bias.data, 0.0)
            else:
                raise RuntimeWarning(f'initialization method [{initial_bn_type}] is not implemented')
        elif hasattr(layer, 'weight'):
            raise RuntimeError(f'[{type(layer)}] with parameters has no initialization method')

    print(f'initialization method [{init_conv_linear_type} + {init_bn_type}], gain is {gain:.2f}')
    fn = functools.partial(init_fn, initial_type=init_conv_linear_type, my_gain=gain, initial_bn_type=init_bn_type)
    net.apply(fn)


def get_network(opt):
    if opt['name'] == 'SFTMD':
        network = SFTMD(opt)
    elif opt['name'] == 'Predictor':
        network = Predictor(opt)
    elif opt['name'] == 'Corrector':
        network = Corrector(opt)
    elif opt['name'] == 'MANet_s1':
        network = MANet_s1(opt)
    elif opt['name'] == 'FFTRCANResUNet':
        network = FFTRCANResUNet(opt)
    elif opt['name'] == 'UNetSR2':
        network = UNetSR2(opt)
    elif opt['name'] == 'DFCAN':
        network = DFCAN(opt)
    elif opt['name'] == 'SFTDFCAN_sequential':
        network = SFTDFCAN_sequential(opt)
    elif opt['name'] == 'SFTDFCAN':
        network = SFTDFCAN(opt)
    else:
        raise NotImplementedError
    if opt['init'] is not None:
        init_weights(network, init_conv_linear_type=opt['init'][0], gain=opt['init'][1], init_bn_type=opt['init'][2])
    return network
