import argparse
import os
import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from scipy.io import savemat

from models import get_model
from datasets import get_dataloader
from utils.universal_util import read_yaml, calculate_PSNR, normalization, PCA_Decoder, nearest_itpl, overlap, \
    draw_text_on_image, pickle_load, calculate_SSIM


def test(opt):
    # pass parameter
    correct_step = opt['testing']['correct_step']
    save_img = opt['testing']['save_img']
    save_mat = opt['testing']['save_mat']
    save_dir = opt['testing']['save_dir']

    # mkdir
    if (save_img or save_mat) and not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # set up data loader
    test_loader = get_dataloader(opt['test_data'])

    # set up model
    F_model = get_model(opt['F_model'])
    P_model = get_model(opt['P_model'])
    C_model = get_model(opt['C_model'])
    pca_decoder = PCA_Decoder(weight=F_model.pca_encoder.weight, mean=F_model.pca_encoder.mean)

    # set up recorder
    pred_kernels = []
    kernel_psnrs = []
    kernel_ssims = []
    kernel_code_psnrs = []
    sr_psnrs = []
    sr_ssims = []
    names = []

    # start testing
    with tqdm(desc=f'Testing', total=len(test_loader.dataset), unit='img') as pbar:
        with torch.no_grad():
            for data in test_loader:
                P_model.feed_data(data)
                P_model.test()
                kernel_code_of_sr = P_model.pred_kernel_code.detach().cpu()
                gt_kernel_code = F_model.pca_encoder(data['kernel'].to(F_model.device))
                for i in range(correct_step):
                    F_model.feed_data({'hr': data['hr'],
                                       'lr': data['lr'],
                                       'kernel_code': kernel_code_of_sr})
                    F_model.test()
                    sr = F_model.sr.detach().cpu()
                    C_model.feed_data({'sr': sr,
                                       'kernel_code_of_sr': kernel_code_of_sr,
                                       'gt_kernel_code': gt_kernel_code})
                    C_model.test()
                    if i <= correct_step - 2:
                        kernel_code_of_sr = C_model.pred_kernel_code.detach().cpu()
                    else:
                        kernel_code_of_sr = kernel_code_of_sr.to(F_model.device)
                gt_kernel = data['kernel'].to(F_model.device).squeeze(0)
                pred_kernel = pca_decoder(kernel_code_of_sr).squeeze(0)
                pred_kernels.append(pred_kernel.detach().cpu().numpy())
                kernel_psnr = calculate_PSNR(pred_kernel, gt_kernel, max_val='auto')
                kernel_psnrs.append(kernel_psnr)
                kernel_ssim = calculate_SSIM(pred_kernel, gt_kernel, rescale=True)
                kernel_ssims.append(kernel_ssim)
                offset = min(torch.min(kernel_code_of_sr).item(), torch.min(gt_kernel_code).item())
                kernel_code_psnr = calculate_PSNR(kernel_code_of_sr - offset, gt_kernel_code - offset, max_val='auto')
                kernel_code_psnrs.append(kernel_code_psnr)
                sr_psnr = calculate_PSNR(F_model.hr, F_model.sr, max_val=1.0)
                sr_psnrs.append(sr_psnr)
                sr_ssim = calculate_SSIM(F_model.hr, F_model.sr)
                sr_ssims.append(sr_ssim)
                names.append(data['name'][0])

                # img marked
                result = torch.cat([nearest_itpl(F_model.lr, F_model.hr.shape[-2:], norm=True),
                                    normalization(F_model.sr),
                                    normalization(F_model.hr)], dim=-1).squeeze(0).squeeze(0)
                show_size = (F_model.hr.shape[-2] // 4, F_model.hr.shape[-1] // 4)
                result = overlap(nearest_itpl(gt_kernel, show_size, norm=True), result, (0, 0))
                result = overlap(nearest_itpl(pred_kernel, show_size, norm=True), result, (show_size[-2], 0))
                result = transforms.ToPILImage()((result * 65535).to(torch.int32))
                font_size = max(F_model.hr.shape[-2] // 25, 16)
                draw_text_on_image(result, data['name'][0], (result.width // 3 * 2, 0), font_size, 65535)
                draw_text_on_image(result, f'PSNR {sr_psnr:5.2f}', (result.width // 3, 0), font_size, 65535)
                draw_text_on_image(result, f'Kernel PSNR {kernel_psnr:5.2f}',
                                   (0, F_model.hr.shape[-2] - 2 * font_size), font_size, 65535)
                draw_text_on_image(result, f'Code PSNR {kernel_code_psnr:5.2f}',
                                   (0, F_model.hr.shape[-2] - font_size), font_size, 65535)

                # img unmarked
                pure = torch.cat([torch.cat([nearest_itpl(F_model.lr, F_model.sr.shape[-2:]),
                                             F_model.sr, F_model.hr], dim=-1),
                                  torch.cat([nearest_itpl(F_model.lr, F_model.sr.shape[-2:], norm=True),
                                             normalization(F_model.sr), normalization(F_model.hr)], dim=-1)],
                                 dim=-2).squeeze(0).squeeze(0)
                pure = transforms.ToPILImage()((pure * 65535).to(torch.int32))

                if save_img:
                    result.save(os.path.join(save_dir, data['name'][0] + '.png'))
                    pure.save(os.path.join(save_dir, data['name'][0] + '_pure.png'))
                pbar.update(1)
    if save_mat:
        savemat(os.path.join(save_dir, 'results.mat'), {'IKC_pred_kernels': pred_kernels,
                                                        'IKC_kernel_psnrs': kernel_psnrs,
                                                        'IKC_kernel_ssims': kernel_ssims,
                                                        'IKC_kernel_code_psnrs': kernel_code_psnrs,
                                                        'IKC_sr_psnrs': sr_psnrs,
                                                        'IKC_sr_ssims': sr_ssims,
                                                        'names': names})
    print(f'avg psnr: sr={np.mean(sr_psnrs):5.2f}, kernel={np.mean(kernel_psnrs):5.2f}, code={np.mean(kernel_code_psnrs):5.2f}')
    print(f'avg ssim: sr={np.mean(sr_ssims):5.3f}, kernel={np.mean(kernel_ssims):5.3f}')


def main():
    """please make sure that the pwd is .../PsfPred rather than .../PsfPred/codes/tests"""
    # set up cmd
    prog = argparse.ArgumentParser()
    prog.add_argument('--opt', type=str, default='./options/test_something.yaml')
    args = prog.parse_args()

    # start train
    test(read_yaml(args.opt))


if __name__ == '__main__':
    main()
