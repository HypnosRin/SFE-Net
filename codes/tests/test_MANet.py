import argparse
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from scipy.io import savemat

from models import get_model
from datasets import get_dataloader
from utils.universal_util import read_yaml, calculate_PSNR, normalization, nearest_itpl, overlap, draw_text_on_image, \
    pickle_load


def test(opt):
    # pass parameter
    save_img = opt['testing']['save_img']
    save_mat = opt['testing']['save_mat']
    save_dir = opt['testing']['save_dir']

    # mkdir
    if (save_img or save_mat) and not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # set up data loader
    test_loader = get_dataloader(opt['test_data'])

    # set up model
    model = get_model(opt['model'])

    # set up recorder
    o = {'MANet_pred_kernels': [], 'names': []}
    pred_kernels = []
    kernel_psnrs = []
    names = []

    # start testing
    with tqdm(desc=f'testing', total=len(test_loader.dataset), unit='img') as pbar:
        with torch.no_grad():
            for data in test_loader:
                model.feed_data(data)
                model.test()

                gt_kernel = model.gt_kernel.squeeze(0)
                # 按LR像素给pred_kernel加权平均，获得预测kernel
                weight = F.interpolate(model.lr, scale_factor=model.network.scale, mode='nearest')
                pred_kernel = (torch.sum(model.pred_kernel * weight.view(1, -1, 1, 1), dim=1) /
                               torch.sum(weight)).squeeze(0)
                pred_kernels.append(pred_kernel.detach().cpu().numpy())
                kernel_psnr = calculate_PSNR(pred_kernel, gt_kernel, max_val='auto')
                kernel_psnrs.append(kernel_psnr)
                names.append(data['name'][0])

                # img marked
                heat_map = model.psnr_heat_map().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                result = torch.cat([nearest_itpl(model.lr, heat_map.shape[-2:], norm=True),
                                    normalization(heat_map),
                                    normalization(data['hr']).to(model.device)], dim=-1).squeeze(0).squeeze(0)
                show_size = (heat_map.shape[-2] // 4, heat_map.shape[-1] // 4)
                result = overlap(nearest_itpl(gt_kernel, show_size, norm=True), result, (0, 0))
                result = overlap(nearest_itpl(pred_kernel, show_size, norm=True), result, (show_size[-2], 0))
                result = transforms.ToPILImage()((result * 65535).to(torch.int32))
                font_size = max(heat_map.shape[-2] // 25, 16)
                draw_text_on_image(result, f'Mean PSNR {kernel_psnr:5.2f}',
                                   (0, heat_map.shape[-2] - 3 * font_size), font_size, 65535)
                draw_text_on_image(result, f'PSNR {torch.min(heat_map).item():5.2f}~{torch.max(heat_map).item():5.2f}',
                                   (0, heat_map.shape[-2] - 2 * font_size), font_size, 65535)
                draw_text_on_image(result, data['name'][0], (0, heat_map.shape[-2] - font_size), font_size, 65535)

                if save_img:
                    result.save(os.path.join(save_dir, data['name'][0] + '.png'))
                pbar.update(1)
    if save_mat:
        savemat(os.path.join(save_dir, 'results.mat'), {'MANet_pred_kernels': pred_kernels,
                                                        'MANet_kernel_psnrs': kernel_psnrs,
                                                        'names': names})
    print(f'avg psnr: kernel={np.mean(kernel_psnrs):5.2f}')


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
