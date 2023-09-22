from torch.utils.data import DataLoader

from datasets.hr_lr_kernel_from_BioSR import HrLrKernelFromBioSR


def get_dataloader(opt):
    # if for test, then batch size should be 1
    assert opt['is_train'] or (not opt['is_train'] and opt['loader_settings']['batch_size'] == 1)
    if opt['name'] == 'HrLrKernelFromBioSR':
        data_set = HrLrKernelFromBioSR(opt)
    else:
        raise NotImplementedError
    data_loader = DataLoader(data_set, **opt['loader_settings'])
    return data_loader
