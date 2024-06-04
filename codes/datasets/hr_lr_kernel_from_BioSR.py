import os
import random
from PIL import Image
import math
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
from scipy.io import savemat, loadmat
import tifffile

from utils.universal_util import random_rotate_crop_flip, add_poisson_gaussian_noise, get_phaseZ
from utils.zernike_psf import ZernikePSFGenerator
from utils.gaussian_kernel import GaussianKernelGenerator


class HrLrKernelFromBioSR(Dataset):
    def __init__(self, opt):
        """
        img names in img_root should be img_xx_y.png, such like img_13_4.png, img_29_3.png
        where y is structure type (CCPs | ER | Microtubules | F-actin), x is index
        """
        super().__init__()
        # general conf
        self.is_train = opt['is_train']
        self.device = torch.device('cpu') if opt['gpu_id'] is None else torch.device('cuda', opt['gpu_id'])
        self.repeat = opt['repeat'] if opt['repeat'] is not None else 1
        # raw hr filter
        self.img_root = opt['img_filter']['img_root']
        self.structure_selected = tuple(opt['img_filter']['structure_selected'])
        self.included_idx = tuple(range(opt['img_filter']['included_idx'][0], opt['img_filter']['included_idx'][1] + 1))
        # hr cropping conf
        self.hr_crop = opt['hr_crop']
        self.hr_size = tuple(opt['hr_crop']['hr_size'])
        # other conf
        self.scale = opt['scale']
        self.img_signal = opt['img_signal']
        # PSF conf
        self.psf_type = opt['psf_settings']['type']
        if self.psf_type == 'Zernike':
            self.phaseZ_settings = opt['psf_settings']['phaseZ']
            opt['psf_settings']['device'] = self.device
            self.psf_gen = ZernikePSFGenerator(opt=opt['psf_settings'])
        elif self.psf_type == 'Gaussian':
            self.psf_gen = GaussianKernelGenerator(opt=opt['psf_settings'])
        else:
            raise NotImplementedError('undefined PSF type')
        # supervised phase for Zernike PSF
        self.sup_phaseZ = opt['sup_phaseZ']
        # padding mode for conv
        self.padding = opt['padding']
        # check
        assert self.hr_size[0] % self.scale == 0 and self.hr_size[1] % self.scale == 0
        # collect all raw hr
        all_gt = os.listdir(self.img_root)
        all_gt.sort()
        self.names = [os.path.splitext(file)[0] for file in all_gt
                      if (file.endswith('.png')
                          and (int(file.split('_')[2].replace('.png', '')) in self.structure_selected)
                          and (int(file[4:6]) in self.included_idx))]
        # generate data in advance in testing mode
        if not self.is_train:
            self.hrs = torch.cat([self.get_aug_hr(i // self.repeat) for i in range(len(self))], dim=-3)
            if self.psf_type == 'Zernike':
                self.test_phaseZs = get_phaseZ(self.phaseZ_settings, batch_size=len(self), device=self.device)
                self.test_kernels = self.psf_gen.generate_PSF(phaseZ=self.test_phaseZs)
            elif self.psf_type == 'Gaussian':
                self.test_kernels = self.psf_gen(batch_size=len(self), tensor=True, random=True)
        # preload data
        if opt['preload_data'] is not None:
            print(f'load data from {opt["preload_data"]}')
            self.load_from_mat(opt['preload_data'])
        else:
            print('generate data on the fly')

    def __len__(self):
        return len(self.names) * self.repeat * (self.hr_crop['scan_shape'][0] * self.hr_crop['scan_shape'][1]
                                                if self.hr_crop['mode'] == 'scan' else 1)

    def get_aug_hr(self, idx):
        """
        :return: GT image, in shape of self.hr_size, with data augmentation
                 Tensor (C, H, W), 0~65535
        """
        name = self.names[idx % len(self.names)] if self.hr_crop['mode'] == 'scan' else self.names[idx]
        img = transforms.ToTensor()(Image.open(os.path.join(self.img_root, name + '.png'))).float()
        if self.hr_crop['mode'] == 'random':
            fill = 0
            while True:  # avoid dark regions
                hr = random_rotate_crop_flip(img, self.hr_size, fill)
                structure = int(name.split('_')[2].replace('.png', ''))
                if structure in (1,) and hr.mean() >= 60:
                    break  # CCPs
                elif structure in (2, 3, 4) and (torch.max(hr) - torch.min(hr)).item() >= 10000:
                    break  # ER, Microtubules, F-actin
                elif structure not in (1, 2, 3, 4):
                    raise NotImplementedError
        elif self.hr_crop['mode'] == 'constant':
            center_pos = self.hr_crop['center_pos']
            bound = ((center_pos[0] - (self.hr_size[0] // 2), center_pos[0] + self.hr_size[0] - (self.hr_size[0] // 2)),
                     (center_pos[1] - (self.hr_size[1] // 2), center_pos[1] + self.hr_size[1] - (self.hr_size[1] // 2)))
            hr = img[:, bound[0][0]:bound[0][1], bound[1][0]:bound[1][1]]
        elif self.hr_crop['mode'] == 'scan':
            start_h = int((img.height - self.hr_size[0]) / self.hr_crop['scan_shape'][0] *
                          ((idx // len(self.names)) % self.hr_crop['scan_shape'][0]))
            start_w = int((img.width - self.hr_size[1]) / self.hr_crop['scan_shape'][1] *
                          ((idx // len(self.names)) // self.hr_crop['scan_shape'][0]))
            hr = img[:, start_h:start_h + self.hr_size[0], start_w:start_w + self.hr_size[1]]
        elif self.hr_crop['mode'] is None:
            hr = img
        else:
            raise NotImplementedError('undefined mode')
        return hr.to(self.device)

    def __getitem__(self, index):
        idx = index // self.repeat
        # get hr & kernel
        if self.is_train:
            hr = self.get_aug_hr(idx)
            if self.psf_type == 'Zernike':
                phaseZ = get_phaseZ(self.phaseZ_settings, batch_size=1, device=self.device)
                kernel = self.psf_gen.generate_PSF(phaseZ=phaseZ)
            elif self.psf_type == 'Gaussian':
                phaseZ = None
                kernel = self.psf_gen(batch_size=1, tensor=True, random=True)
            else:
                raise NotImplementedError('undefined PSF type')
        else:
            hr = self.hrs[index:index + 1, :, :]
            if self.psf_type == 'Zernike':
                phaseZ = self.test_phaseZs[index:index + 1, :]
                kernel = self.test_kernels[index:index + 1, :, :]
            elif self.psf_type == 'Gaussian':
                phaseZ = None
                kernel = self.test_kernels[index:index + 1, :, :]
            else:
                raise NotImplementedError('undefined PSF type')
        assert kernel.shape[-2] % 2 == 1 and kernel.shape[-1] % 2 == 1, 'kernel shape should be odd'
        # do conv
        pad = (kernel.shape[-2] // 2,) * 2 + (kernel.shape[-1] // 2,) * 2
        if self.padding['mode'] == "circular":
            lr = F.conv2d(F.pad(hr.unsqueeze(0), pad=pad, mode=self.padding['mode']), kernel.unsqueeze(0)).squeeze(0)
        else:
            lr = F.conv2d(F.pad(hr.unsqueeze(0), pad=pad, mode=self.padding['mode'], value=self.padding['value']),
                          kernel.unsqueeze(0)).squeeze(0)
        # add noise
        img_signal = 10.0 ** random.uniform(math.log10(self.img_signal[0]), math.log10(self.img_signal[-1]))
        # TODO delete
        # origin_lr = lr / 65535.0
        lr = add_poisson_gaussian_noise(lr, level=img_signal)
        # down sample
        lr = lr[:, ::self.scale, ::self.scale]
        # do supervised phase
        if self.sup_phaseZ == 'all' or self.psf_type != 'Zernike':
            pass
        else:
            cut_phaseZ = torch.zeros(size=phaseZ.shape, dtype=phaseZ.dtype, device=phaseZ.device)
            cut_phaseZ[..., self.sup_phaseZ[0]:self.sup_phaseZ[-1] + 1] = \
                phaseZ[..., self.sup_phaseZ[0]:self.sup_phaseZ[-1] + 1]
            kernel = self.psf_gen.generate_PSF(phaseZ=cut_phaseZ)
        # modify name
        if self.hr_crop['mode'] == 'scan':
            name = self.names[idx % len(self.names)] + f'_part{idx // len(self.names)}' + \
                   f'_{(index % self.repeat) + 1}'
        else:
            name = self.names[idx] + f'_{(index % self.repeat) + 1}'
        # set to [0, 1]
        hr = hr / 65535.0
        lr = lr / 65535.0
        # return
        if self.psf_type == 'Zernike':
            return {'hr': hr,  # (C, H, W), [0, 1]
                    # 'origin_lr': origin_lr,  # TODO delete
                    'lr': lr,  # (C, H, W), [0, 1]
                    'kernel': kernel.squeeze(0),  # (H, W), sum up to 1.0
                    'name': name,  # str, without postfix '.png'
                    'phaseZ': phaseZ,  # (1, 25)
                    'img_signal': img_signal}  # float
        elif self.psf_type == 'Gaussian':
            return {'hr': hr,  # (C, H, W), [0, 1]
                    'lr': lr,  # (C, H, W), [0, 1]
                    'kernel': kernel.squeeze(0),  # (H, W), sum up to 1.0
                    'name': name,  # str, without postfix '.png'
                    'img_signal': img_signal}  # float

    def save_as_mat(self, save_path):
        l = len(self)
        data = [self[i] for i in range(l)]
        hrs = [x['hr'].squeeze(0).squeeze(0).cpu().numpy() for x in data]
        lrs = [x['lr'].squeeze(0).squeeze(0).cpu().numpy() for x in data]
        gt_kernels = [x['kernel'].squeeze(0).cpu().numpy() for x in data]
        phaseZ = [x['phaseZ'].squeeze(0).cpu().numpy() for x in data]
        names = [x['name'] for x in data]

        savemat(save_path, {'hrs': hrs, 'lrs': lrs, 'gt_kernels': gt_kernels, 'names': names, 'phaseZ': phaseZ})

    def save_img(self, save_path):
        import skimage.io as io
        from utils.universal_util import normalization
        import numpy as np

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data = [self[i] for i in range(len(self))]
        for d in data:
            # tifffile.imwrite(os.path.join(save_path, d['name'] + "_lr_" + str(d['img_signal']) + "_phase3_" + str(
            #     d['phaseZ'][0,8].cpu().numpy()) + ".tif"), d['lr'].squeeze(0).cpu().numpy())
            io.imsave(os.path.join(save_path, d['name'] + "_lr_origin.png"),
                      np.uint16(normalization(d['origin_lr'].squeeze(0).cpu()).numpy() * 65535))
            io.imsave(os.path.join(save_path, d['name'] + "_lr.png"),
                      np.uint16(normalization(d['lr'].squeeze(0).cpu()).numpy() * 65535))
            io.imsave(os.path.join(save_path, d['name'] + "_hr.png"),
                      np.uint16(normalization(d['hr'].squeeze(0).cpu()).numpy() * 65535))
            io.imsave(os.path.join(save_path, d['name'] + "_k_gray.png"),
                      np.uint16(normalization(d['kernel'].squeeze(0).cpu()).numpy() * 65535))

            # tifffile.imwrite(os.path.join(save_path, d['name'] + "_lr_origin.tif"),
            #                  d['origin_lr'].squeeze(0).cpu().numpy())  # TODO delete
            # tifffile.imwrite(os.path.join(save_path, d['name'] + "_lr.tif"), d['lr'].squeeze(0).cpu().numpy())
            # tifffile.imwrite(os.path.join(save_path, d['name'] + "_hr.tif"), d['hr'].squeeze(0).cpu().numpy())
            # tifffile.imwrite(os.path.join(save_path, d['name'] + "_k_gray.tif"), d['kernel'].squeeze(0).cpu().numpy())

    def load_from_mat(self, load_path):
        data = loadmat(load_path)
        self.hrs = torch.from_numpy(data['hrs']) * 65535.0
        self.test_kernels = torch.from_numpy(data['gt_kernels'])


def main():
    # test data set options
    # opt = {'name': 'HrLrKernelFromBioSR',
    #        'is_train': True,
    #        'gpu_id': None,  # None for cpu
    #        'repeat': None,
    #        'img_filter': {'img_root': '../../../BioDatasets/BioSR/Mixed',
    #                       'structure_selected': [1, 2, 3, 4],
    #                       'included_idx': [11, 100]},
    #        'hr_crop': {'mode': 'random',  # random | constant | scan
    #                    'center_pos': [-1, -1],  # [H, W], for constant
    #                    'scan_shape': [-1, -1],  # [H, W], for scan
    #                    'hr_size': [264, 264]},  # [H, W]
    #        'scale': 2,
    #        'img_signal': [100, 1000],
    #        'psf_settings': {'kernel_size': 33,
    #                         'NA': 1.35,
    #                         'Lambda': 0.525,
    #                         'RefractiveIndex': 1.33,
    #                         'SigmaX': 2.0,
    #                         'SigmaY': 2.0,
    #                         'Pixelsize': 0.0313,
    #                         'nMed': 1.33,
    #                         'phaseZ': {'idx_start': 4,
    #                                    'num_idx': 15,
    #                                    'mode': 'gaussian',  # gaussian | uniform
    #                                    'std': 0.125,  # for gaussian
    #                                    'bound': 1.0}},  # for gaussian and uniform
    #        'sup_phaseZ': 'all',  # all | [begin, end]
    #        'padding': {'mode': 'circular',  # constant | reflect | replicate | circular
    #                    'value': -1},  # for constant mode
    #        'loader_settings': {'batch_size': 4,
    #                            'shuffle': True,
    #                            'num_workers': 3,
    #                            'pin_memory': False,
    #                            'drop_last': True}}
    # data_set = HrLrKernelFromBioSR(opt)
    # data_loader = torch.utils.data.DataLoader(data_set, **opt['loader_settings'])
    # print(len(data_loader), len(data_set),
    #       data_set[0]['hr'].shape, data_set[0]['lr'].shape,
    #       data_set[0]['kernel'].shape, data_set[0]['name'],
    #       data_set[0]['phaseZ'].shape, data_set[0]['img_signal'])

    # save data set for test diff zernike
    opt = {'name': 'HrLrKernelFromBioSR',
           'is_train': False,
           'preload_data': None,
           'gpu_id': None,  # None for cpu
           'repeat': 3,
           'img_filter': {'img_root': './BioDataset/Mixed',
                          'structure_selected': [1,2,3],  # 1 CCPs 2 ER 3 MTs 4 F-actin
                          'included_idx': [1, 10]},  # 1-10 for test; 11-100 for train
           'hr_crop': {'mode': 'random',  # random | constant | scan
                       'center_pos': [-1, -1],  # [H, W], for constant
                       'scan_shape': [-1, -1],  # [H, W], for scan
                       'hr_size': [264, 264]},  # [H, W] 264 for default sr test; 128 for default train
           'scale': 2,
           'img_signal': [120, 400],
           'psf_settings': {
               'type': 'Zernike',
               'kernel_size': 33,
               'sigma': 2.6,
               'sigma_min': 0.2,
               'sigma_max': 4.0,
               'prob_isotropic': 0.0,
               'scale': 2,

               'NA': 1.35,
               'Lambda': 0.525,
               'RefractiveIndex': 1.33,
               'SigmaX': 2.0,
               'SigmaY': 2.0,
               'Pixelsize': 0.0313,
               'nMed': 1.33,
               'phaseZ': {'idx_start': [3,6],
                          'num_idx': [1,2],
                          'mode': 'uniform',  # gaussian | uniform
                          'std': 0.125,  # for gaussian
                          'bound': [0.25, 0.4]}},  # for gaussian and uniform
           'sup_phaseZ': 'all',  # all | [begin, end]
           'padding': {'mode': 'circular',  # constant | reflect | replicate | circular
                       'value': -1},  # for constant mode
           'loader_settings': {'batch_size': 4,
                               'shuffle': False,
                               'num_workers': 0,
                               'pin_memory': False,
                               'drop_last': False}}

    # save data set for train diff zernike

    data_set = HrLrKernelFromBioSR(opt)
    data_loader = torch.utils.data.DataLoader(data_set, **opt['loader_settings'])
    print(len(data_loader), len(data_set),
          data_set[0]['hr'].shape, data_set[0]['lr'].shape,
          data_set[0]['kernel'].shape, data_set[0]['name'],
          data_set[0]['phaseZ'].shape, data_set[0]['img_signal'])
    data_set.save_as_mat('./data/test_loaders_test_version/zernike-3-6-7s.mat')
    # data_set.save_img('./data/test_loaders_test_version/zernike-6-7s')


if __name__ == '__main__':
    main()
