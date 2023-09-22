from models.IKC import F_Model, P_Model, C_Model
from models.MANet import MANet_Model
from models.unet_based import UNetBased_Model
from models.unet_sr import UNetSR_Model
from models.DFCAN import DFCAN
from models.SFTDFCAN import SFTDFCAN


def get_model(opt):
    if opt['name'] is None:
        return None
    elif opt['name'] == 'F_Model':
        return F_Model(opt)
    elif opt['name'] == 'P_Model':
        return P_Model(opt)
    elif opt['name'] == 'C_Model':
        return C_Model(opt)
    elif opt['name'] == 'MANet':
        return MANet_Model(opt)
    elif opt['name'] == 'UNetBased':
        return UNetBased_Model(opt)
    elif opt['name'] == 'UNetSR':
        return UNetSR_Model(opt)
    elif opt['name'] == 'DFCAN':
        return DFCAN(opt)
    elif opt['name'] == 'SFTDFCAN':
        return SFTDFCAN(opt)
    else:
        raise NotImplementedError
