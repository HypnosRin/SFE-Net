import math
import torch
import numpy as np
import scipy
from torchvision import transforms
from tqdm import tqdm
from utils.universal_util import rectangular_closure


class ZernikePSFGenerator(object):
    """
    只适用于z=0的PSF，每套phaseZ参数只生成单张PSF，只取多项式的前25项
    Index: Wyant. Meaning of each phaseZ element:
    0 = Piston
    1, 2 = Horizontal / Verticaltilt
    3 = Defocus
    4, 5 = Oblique / Vertical astigmatism
    6, 7 = Horizontal / Vertical coma
    8 = Primary spherical
    9, 10 = Oblique / Vertical trefoil
    11, 12 = Vertical / Oblique secondary astigmatism
    """

    def __init__(self, opt):
        super().__init__()
        self.device = opt['device']
        self.kernel_size = opt['kernel_size']
        self.NA = opt['NA']
        self.Lambda = opt['Lambda']
        self.RefractiveIndex = opt['RefractiveIndex']
        self.SigmaX = opt['SigmaX']
        self.SigmaY = opt['SigmaY']
        self.Pixelsize = opt['Pixelsize']
        self.nMed = opt['nMed']
        PSFsize = 128
        assert PSFsize % 2 == 0
        self.PSFsize = PSFsize

        pos = torch.arange(-PSFsize / 2, PSFsize / 2, 1)
        Y, X = torch.meshgrid(pos, pos)
        k_r = (torch.sqrt(X * X + Y * Y)) / (PSFsize * self.Pixelsize)
        Phi = torch.arctan2(Y, X)
        NA_constrain = torch.less(k_r, self.NA / self.Lambda)
        NA_closure = rectangular_closure(NA_constrain)
        # k_z = torch.sqrt(((self.RefractiveIndex / self.Lambda) ** 2 - k_r * k_r) * NA_constrain)
        # sin_theta3 = k_r * self.Lambda / self.RefractiveIndex
        # sin_theta1 = self.RefractiveIndex / self.nMed * sin_theta3
        # Cos1 = torch.sqrt((1 - sin_theta1 * sin_theta1).type(torch.complex64))
        # Cos3 = torch.sqrt((1 - sin_theta3 * sin_theta3).type(torch.complex64))

        # ZN = int(math.sqrt(len(phaseZ)) - 1)
        ZN = 4
        Wnmax = ZN
        Wlmax = (ZN + 1) ** 2 - 1
        WZc = [[]] * (Wnmax + 1)
        for n in range(0, Wnmax + 1):
            WZc[n] = [[]] * (n + 1)
            for m in range(0, n + 1):
                WZc[n][m] = [[]] * (n - m + 1)
                for k in range(0, n - m + 1):
                    WZc[n][m][k] = \
                        ((-1) ** k) * np.prod(np.arange(n - m - k + 1, 2 * n - m - k + 1, 1)) / \
                        (math.factorial(k) * math.factorial(n - k))

        rho = k_r * NA_constrain / (self.NA / self.Lambda)
        theta = Phi * NA_constrain
        assert Wnmax < len(WZc)

        rhos = torch.empty((PSFsize, PSFsize, 2 * Wnmax + 1))
        rhos[:, :, 0] = 1
        for i in range(1, 2 * Wnmax + 1):
            rhos[:, :, i] = rhos[:, :, i - 1] * rho

        thetas = torch.empty((PSFsize, PSFsize, Wnmax))
        thetas[:, :, 0] = theta
        for i in range(1, Wnmax):
            thetas[:, :, i] = thetas[:, :, i - 1] + theta

        self.Z = torch.ones((PSFsize, PSFsize, 1 + Wlmax))
        l = 1
        self.Z[:, :, l - 1] = NA_constrain
        for n in range(1, Wnmax + 1):
            for m in range(n, -1, -1):
                c = WZc[n][m]
                cc = torch.ones((PSFsize, PSFsize, len(c)))
                for i in range(0, len(c)):
                    cc[:, :, i] = c[i]
                my_slice = list(range(2 * n - m, m - 2, -2))
                p = NA_constrain * torch.sum(cc * rhos[:, :, my_slice], dim=2)
                if m == 0:
                    l += 1
                    self.Z[:, :, l - 1] = p
                else:
                    l += 1
                    self.Z[:, :, l - 1] = p * torch.cos(thetas[:, :, m - 1])
                    l += 1
                    self.Z[:, :, l - 1] = p * torch.sin(thetas[:, :, m - 1])
                if l > Wlmax:
                    break
        for i in range(1 + Wlmax):
            Z_closure = rectangular_closure(self.Z[:, :, i] != 0.0)
            assert NA_closure == Z_closure
        self.mask_closure = NA_closure
        assert self.mask_closure[1] - self.mask_closure[0] + 1 == self.mask_closure[3] - self.mask_closure[2] + 1
        self.mask_l = self.mask_closure[1] - self.mask_closure[0] + 1

        self.magZ = torch.zeros((1, 25))
        self.magZ[:, 0] = 1.0
        self.pupil_mag = torch.zeros((1, PSFsize, PSFsize))
        for k in range(0, 25):
            self.pupil_mag += self.Z[:, :, k].unsqueeze(0) * self.magZ[:, k].unsqueeze(-1).unsqueeze(-1)
        self.Z = self.Z.to(self.device)
        self.pupil_mag = self.pupil_mag.to(self.device)

        realsize0_gen = math.floor(self.kernel_size / 2)
        realsize1_gen = math.ceil(self.kernel_size / 2)
        self.startx_gen = -realsize0_gen + PSFsize // 2
        self.endx_gen = realsize1_gen + PSFsize // 2
        self.starty_gen = -realsize0_gen + PSFsize // 2
        self.endy_gen = realsize1_gen + PSFsize // 2

        cropsize = min(29, self.kernel_size)
        sigmaXr = 1 / 2 / torch.pi / self.SigmaX
        sigmaYr = 1 / 2 / torch.pi / self.SigmaY
        pos = torch.arange(-self.kernel_size / 2, self.kernel_size / 2, 1)
        Y, X = torch.meshgrid(pos, pos)
        xx, yy = X * self.Pixelsize, Y * self.Pixelsize
        gauss_r = 2 * torch.pi * self.SigmaX * self.SigmaY * \
                  torch.exp(-(xx * xx) / 2 / (sigmaXr ** 2)) * \
                  torch.exp(-(yy * yy) / 2 / (sigmaYr ** 2))
        realsize0_scale = math.floor(cropsize / 2)
        realsize1_scale = math.ceil(cropsize / 2)
        startx_scale = math.floor(-realsize0_scale + self.kernel_size / 2 + 1)
        endx_scale = math.floor(realsize1_scale + self.kernel_size / 2)
        starty_scale = math.floor(-realsize0_scale + self.kernel_size / 2 + 1)
        endy_scale = math.floor(realsize1_scale + self.kernel_size / 2)
        gauss_r = gauss_r[startx_scale - 1:endx_scale, starty_scale - 1:endy_scale] * (self.Pixelsize ** 2)
        self.gauss_r = torch.flip(gauss_r, dims=(0, 1)).to(self.device)
        pad_H, pad_W = self.gauss_r.shape[-2] // 2, self.gauss_r.shape[-1] // 2
        self.psf_padding = (pad_H, pad_H, pad_W, pad_W)

    def generate_PSF(self, phaseZ, blur=True):
        """phaseZ shape (N, 25), return kernels shape (N, kernel_size, kernel_size)"""
        return self.pupil_phase_to_PSF(self.phaseZ_to_pupil_phase(phaseZ), blur=blur)

    def phaseZ_to_pupil_phase(self, phaseZ):
        """phaseZ shape (N, 25), return pupil_phase shape (N, PSFsize, PSFsize)"""
        assert phaseZ.device == self.Z.device
        pupil_phase = torch.zeros((phaseZ.shape[0], self.PSFsize, self.PSFsize)).to(self.device)
        for k in range(phaseZ.shape[-1]):
            pupil_phase += self.Z[:, :, k].unsqueeze(0) * phaseZ[:, k].unsqueeze(-1).unsqueeze(-1)
        # 由于self.Z的每一阶的mask外都为0，所以pupil_phase的mask外也都为0
        return pupil_phase

    def mask_pupil_phase(self, pupil_phase):
        """pupil_phase shape (..., PSFsize, PSFsize), return pupil_phase shape (..., mask_l, mask_l)"""
        assert pupil_phase.shape[-2:] == (self.PSFsize, self.PSFsize)
        return pupil_phase[..., self.mask_closure[0]:self.mask_closure[1] + 1,
               self.mask_closure[2]:self.mask_closure[3] + 1]

    def pad_pupil_phase(self, pupil_phase):
        """pupil_phase shape (..., mask_l, mask_l), return pupil_phase shape (..., PSFsize, PSFsize)"""
        assert pupil_phase.shape[-2:] == (self.mask_l, self.mask_l)
        pad = (self.mask_closure[2], self.PSFsize - 1 - self.mask_closure[3], self.mask_closure[0],
               self.PSFsize - 1 - self.mask_closure[1])
        return torch.nn.functional.pad(pupil_phase, pad=pad, mode='constant', value=0.0)

    def pupil_phase_to_PSF(self, pupil_phase, blur=True):
        """pupil_phase shape (N, PSFsize, PSFsize), return kernels shape (N, kernel_size, kernel_size)"""
        # self.pupil_mag是一个位于tensor中心的圆形mask
        pupil_complex = self.pupil_mag * torch.exp(2j * torch.pi * pupil_phase)
        # fftshift将零频分量移到tensor中心
        psfs = torch.abs(torch.fft.fftshift(torch.fft.fft2(pupil_complex))) ** 2
        # 裁剪成kernel_size
        psfs = psfs[:, self.startx_gen:self.endx_gen, self.starty_gen:self.endy_gen] / (self.PSFsize ** 2)
        # 卷积模糊核
        if blur:
            padded_psfs = torch.nn.functional.pad(psfs, pad=self.psf_padding, mode='constant', value=0.0).unsqueeze(1)
            blur_kernel = self.gauss_r.unsqueeze(0).unsqueeze(0)
            scaled_psfs = torch.nn.functional.conv2d(padded_psfs, blur_kernel).squeeze(1)
        else:
            scaled_psfs = psfs
        return scaled_psfs / torch.sum(scaled_psfs, dim=(-2, -1), keepdim=True)


def main():
    opt = {'name': 'HrLrKernelFromBioSR',
           'is_train': False,
           'preload_data': None,
           'gpu_id': None,  # None for cpu
           'repeat': 3,
           'img_filter': {'img_root': '../../BioDataset/fig',
                          'structure_selected': [3],  # 1 CCPs 2 ER 3 MTs 4 F-actin
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
               'phaseZ': {'idx_start': [6],
                          'num_idx': [2],
                          'mode': 'uniform',  # gaussian | uniform
                          'std': 0.125,  # for gaussian
                          'bound': [1.0]}},  # for gaussian and uniform
           'sup_phaseZ': 'all',  # all | [begin, end]
           'padding': {'mode': 'circular',  # constant | reflect | replicate | circular
                       'value': -1},  # for constant mode
           'loader_settings': {'batch_size': 4,
                               'shuffle': False,
                               'num_workers': 0,
                               'pin_memory': False,
                               'drop_last': False}}

    device = torch.device('cpu') if opt['gpu_id'] is None else torch.device('cuda', opt['gpu_id'])
    opt['psf_settings']['device'] = device
    psf_gen = ZernikePSFGenerator(opt=opt['psf_settings'])

    # import pylab
    # phaseZ = torch.zeros((1,25))
    # pylab.figure()
    # c = 1
    # for i in range(9):
    #     for j in range(9):
    #         phaseZ[0,6] = -0.5 + i / 8
    #         phaseZ[0,7] = -0.5 + j / 8
    #         # print(str(i + 1) + str(j + 1) + str(c))
    #         pylab.subplot(9,9,c)
    #         pylab.imshow(psf_gen.generate_PSF(phaseZ)[0],'gray')
    #         c+=1

    import pylab
    phaseZ = torch.tensor([[0,0,0,0,
         0,
    0.0207,
    0.0368,
    0.0119,
    0.0358,
   -0.0306,
   -0.0401,
   -0.0105,
   -0.1731,
   -0.0410,
   -0.0690,
   -0.0194,
    0.0149,
   -0.1078,
   -0.0038,
                           0,0,0,0,0,0]])
    pylab.figure()
    pylab.imshow(torch.abs(psf_gen.phaseZ_to_pupil_phase(phaseZ)[0]).numpy(),'gray')

    pylab.figure()
    pylab.imshow(psf_gen.generate_PSF(phaseZ)[0],'gray')

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # kernel_num = 1000
    # kernel_size = 33
    # # opt = {'device': device,
    # #        'kernel_size': 55,
    # #        'NA': 0.6,
    # #        'Lambda': 0.605,
    # #        'RefractiveIndex': 1.33,
    # #        'SigmaX': 2.0,
    # #        'SigmaY': 2.0,
    # #        'Pixelsize': 0.0500,
    # #        'nMed': 1.33}
    # opt = {'device': device,
    #        'kernel_size': 33,
    #        'NA': 1.35,
    #        'Lambda': 0.525,
    #        'RefractiveIndex': 1.33,
    #        'SigmaX': 2.0,
    #        'SigmaY': 2.0,
    #        'Pixelsize': 0.0313,
    #        'nMed': 1.33}
    # psf_gen = ZernikePSFGenerator(opt=opt)
    # is_show = False
    #
    # phaseZ = torch.zeros(25).to(device)
    #
    # PSFs = torch.empty((kernel_num, kernel_size, kernel_size)).to(device)
    # Zernike_phases = torch.empty(kernel_num, 25).to(device)
    # Zernike_mags = torch.empty(kernel_num, 25).to(device)
    # NAs = torch.empty(kernel_num).to(device)
    # lambdas = torch.empty(kernel_num).to(device)
    # refractive_indexes = torch.empty(kernel_num).to(device)
    # sigmaXs = torch.empty(kernel_num).to(device)
    # sigmaYs = torch.empty(kernel_num).to(device)
    # pixel_sizes = torch.empty(kernel_num).to(device)
    # nMeds = torch.empty(kernel_num).to(device)
    #
    # with tqdm(desc=f'Generating', total=kernel_num, unit='kernel') as pbar:
    #     for iteration in range(1, kernel_num + 1):
    #         phaseZ[...] = 0.0
    #         num_p = 11
    #         phaseZ[..., 4:4 + num_p] = torch.normal(mean=0.0, std=0.125, size=(num_p,), device=device)
    #         phaseZ = torch.clamp(phaseZ, min=-1.0, max=1.0)
    #
    #         PSFs[iteration - 1, :, :] = psf_gen.generate_PSF(phaseZ=phaseZ.unsqueeze(0)).squeeze(0)
    #
    #         if is_show and iteration > 100:
    #             kernels = [
    #                 (PSFs[i, :, :] - torch.min(PSFs[i, :, :])) / (torch.max(PSFs[i, :, :]) - torch.min(PSFs[i, :, :]))
    #                 for i in range(100)]
    #             kernels = torch.cat([torch.cat([kernels[i * 10 + j] for j in range(10)], dim=1) for i in range(10)],
    #                                 dim=0)
    #             transforms.ToPILImage()(kernels).save('./kernels.png')
    #             exit()
    #
    #         Zernike_phases[iteration - 1, :] = phaseZ
    #         Zernike_mags[iteration - 1, :] = psf_gen.magZ.squeeze(0)
    #         NAs[iteration - 1] = psf_gen.NA
    #         lambdas[iteration - 1] = psf_gen.Lambda
    #         refractive_indexes[iteration - 1] = psf_gen.RefractiveIndex
    #         sigmaXs[iteration - 1] = psf_gen.SigmaX
    #         sigmaYs[iteration - 1] = psf_gen.SigmaY
    #         pixel_sizes[iteration - 1] = psf_gen.Pixelsize
    #         nMeds[iteration - 1] = psf_gen.nMed
    #
    #         pbar.update(1)
    #
    # tensor2ndarray = lambda x: x.cpu().numpy().astype(np.float32)
    # PSFs = tensor2ndarray(PSFs)
    # Zernike_phases = tensor2ndarray(Zernike_phases)
    # Zernike_mags = tensor2ndarray(Zernike_mags)
    # NAs = tensor2ndarray(NAs)
    # lambdas = tensor2ndarray(lambdas)
    # refractive_indexes = tensor2ndarray(refractive_indexes)
    # sigmaXs = tensor2ndarray(sigmaXs)
    # sigmaYs = tensor2ndarray(sigmaYs)
    # pixel_sizes = tensor2ndarray(pixel_sizes)
    # nMeds = tensor2ndarray(nMeds)
    #
    # scipy.io.savemat(f'./datasets/ForBiosrPSF_{kernel_num}.mat',
    #                  {'PSFs': PSFs, 'Zernike_phases': Zernike_phases, 'Zernike_mags': Zernike_mags,
    #                   'NAs': NAs, 'lambdas': lambdas, 'refractive_indexes': refractive_indexes,
    #                   'sigmaXs': sigmaXs, 'sigmaYs': sigmaYs, 'pixel_sizes': pixel_sizes, 'nMeds': nMeds})


if __name__ == '__main__':
    main()
