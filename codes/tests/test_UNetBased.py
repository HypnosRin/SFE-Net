import argparse
import os
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from scipy.io import savemat

from models import get_model
from datasets import get_dataloader
from utils.universal_util import read_yaml, calculate_PSNR, normalization, nearest_itpl, overlap, draw_text_on_image, \
    pickle_load

import time

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
    pred_kernels = []
    kernel_psnrs = []
    names = []


    iter = 0
    times = []

    # start testing
    with tqdm(desc=f'testing', total=len(test_loader.dataset), unit='img') as pbar:
        with torch.no_grad():
            for data in test_loader:
                iter += 1
                if iter % 50 == 0:
                    times.append(time.perf_counter())

                model.feed_data(data)
                model.test()

                gt_kernel = model.gt_kernel.squeeze(0)
                pred_kernel = model.pred_kernel.squeeze(0).squeeze(0)
                pred_kernels.append(pred_kernel.detach().cpu().numpy())
                kernel_psnr = calculate_PSNR(pred_kernel, gt_kernel, max_val='auto')
                kernel_psnrs.append(kernel_psnr)
                names.append(data['name'][0])

                # img marked
                result = normalization(model.lr).squeeze(0).squeeze(0)
                show_size = (model.lr.shape[-2] // 4, model.lr.shape[-1] // 4)
                result = overlap(nearest_itpl(gt_kernel, show_size, norm=True), result, (0, 0))
                result = overlap(nearest_itpl(pred_kernel, show_size, norm=True), result, (show_size[-2], 0))
                result = transforms.ToPILImage()((result * 65535).to(torch.int32))
                font_size = max(model.lr.shape[-2] // 25, 16)
                draw_text_on_image(result, f'PSNR {kernel_psnr:5.2f}',
                                   (0, model.lr.shape[-2] - 2 * font_size), font_size, 65535)
                draw_text_on_image(result, data['name'][0], (0, model.lr.shape[-2] - font_size), font_size, 65535)

                if save_img:
                    result.save(os.path.join(save_dir, data['name'][0] + '.png'))
                pbar.update(1)
    if save_mat:
        savemat(os.path.join(save_dir, 'results.mat'), {'UNetBased_pred_kernels': pred_kernels,
                                                        'UNetBased_kernel_psnrs': kernel_psnrs,
                                                        'names': names})
    print(f'avg psnr: kernel={np.mean(kernel_psnrs):5.2f}')

    duration = []
    for i in range(1,len(times)):
        duration.append(times[i]- times[i-1])

    print("predict time: mean {}s; std {}s; min {}s; max {}s".format(np.mean(duration), np.std(duration),
                                                                     np.min(duration), np.max(duration)))




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
