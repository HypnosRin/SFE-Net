import torch
from tqdm import tqdm

from utils.zernike_psf import ZernikePSFGenerator
from utils.gaussian_kernel import GaussianKernelGenerator
from utils.universal_util import get_phaseZ, pickle_dump, PCA_Encoder, PCA_Decoder, calculate_PSNR, normalization, \
    save_yaml


def PCA(x, h=2):
    """
    :param x: (batch_size, num_feature)
    :param h:
    :return: (num_feature, h)
    """
    x_mean = torch.mean(x, dim=0, keepdim=True)
    x = x - x_mean
    U, S, V = torch.svd(torch.t(x))
    return U[:, :h]  # PCA matrix


def main():
    # params
    opt = {'type': 'Gaussian',  # Zernike | Gaussian
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
           'phaseZ': {'idx_start': 4,
                      'num_idx': 15,
                      'mode': 'gaussian',
                      'std': 0.125,
                      'bound': 1.0},
           'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')}
    total = 80000
    batch = 1000
    norm = False
    # set generator
    if opt['type'] == 'Zernike':
        psf_gen = ZernikePSFGenerator(opt)
    elif opt['type'] == 'Gaussian':
        psf_gen = GaussianKernelGenerator(opt)
    else:
        raise NotImplementedError('undefined type')
    # generate PSF
    sample = []
    other = []
    with tqdm(desc=f'generating...', total=total, unit='psf') as pbar:
        for _ in range(total // batch):
            if isinstance(psf_gen, ZernikePSFGenerator):
                s = psf_gen.generate_PSF(phaseZ=get_phaseZ(opt['phaseZ'], batch_size=batch, device=opt['device']))
                o = psf_gen.generate_PSF(phaseZ=get_phaseZ(opt['phaseZ'], batch_size=batch, device=opt['device']))
            elif isinstance(psf_gen, GaussianKernelGenerator):
                s = psf_gen(batch_size=batch, tensor=True, random=True)
                o = psf_gen(batch_size=batch, tensor=True, random=True)
            else:
                raise NotImplementedError('undefined type')
            if norm:
                s, o = normalization(s, batch=batch > 1), normalization(o, batch=batch > 1)
            sample.append(s)
            other.append(o)
            pbar.update(batch)
    sample = torch.cat(sample, dim=0)
    other = torch.cat(other, dim=0)
    flat = sample.view(sample.shape[0], -1)
    # do pca
    do_pca(flat, other)
    # back up conf
    opt['device'] = str(opt['device'])
    save_yaml(opt=opt, yaml_path='./psf_settings.yaml')


def do_pca(flat, other):
    """
    do pca and validate PSNR
    :param flat: (batch_size, num_features)
    :param other: (batch_size, kernel_size, kernel_size)
    :return: None
    """
    pca_mean = torch.mean(flat, dim=0, keepdim=True)
    pickle_dump(pca_mean.float().cpu(), './pca_mean.pth')
    for h in (10, 65, 92, 112):
        pca_matrix = PCA(flat, h=h)
        # h的值通过试探来确定，一般设定主成分贡献占比大于99%
        # (2023/1/11) 实验发现在当前配置下，h=65时占比99%(PSNR≈68)，h=92时占比99.9%(PSNR≈88)，h=112时占比99.99%(PSNR≈100)
        pickle_dump(pca_matrix.float().cpu(), f'./pca_matrix{h}.pth')
        zip_unzip = PCA_Decoder(pca_matrix, pca_mean)(PCA_Encoder(pca_matrix, pca_mean)(other))
        print(f'h={h}, PSNR={calculate_PSNR(other, zip_unzip, max_val="auto")}')


if __name__ == '__main__':
    main()
