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
    pickle_load, calculate_SSIM


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
    sr_psnrs = []
    sr_ssims = []
    names = []

    # start testing
    with tqdm(desc=f'testing', total=len(test_loader.dataset), unit='img') as pbar:
        with torch.no_grad():
            for data in test_loader:
                model.feed_data(data)
                model.test()

                sr_psnr = calculate_PSNR(model.hr, model.sr, max_val=1.0)
                sr_psnrs.append(sr_psnr)
                sr_ssim = calculate_SSIM(model.hr, model.sr)
                sr_ssims.append(sr_ssim)
                names.append(data['name'][0])

                # img marked
                result = torch.cat([nearest_itpl(model.lr, model.sr.shape[-2:], norm=True), normalization(model.sr),
                                    normalization(model.hr)], dim=-1).squeeze(0).squeeze(0)
                show_size = (model.hr.shape[-2] // 4, model.hr.shape[-1] // 4)
                result = overlap(nearest_itpl(data['kernel'].squeeze(0), show_size, norm=True), result, (0, 0))
                result = transforms.ToPILImage()((result * 65535).to(torch.int32))
                font_size = max(model.hr.shape[-2] // 25, 16)
                draw_text_on_image(result, f'PSNR {sr_psnr:5.2f}', (result.width // 3, 0), font_size, 65535)
                draw_text_on_image(result, data['name'][0], (result.width // 3 * 2, 0), font_size, 65535)

                # img unmarked
                pure = torch.cat([torch.cat([nearest_itpl(model.lr, model.sr.shape[-2:]),
                                             model.sr, model.hr], dim=-1),
                                  torch.cat([nearest_itpl(model.lr, model.sr.shape[-2:], norm=True),
                                             normalization(model.sr), normalization(model.hr)], dim=-1)],
                                 dim=-2).squeeze(0).squeeze(0)
                pure = transforms.ToPILImage()((pure * 65535).to(torch.int32))

                if save_img:
                    result.save(os.path.join(save_dir, data['name'][0] + '.png'))
                    pure.save(os.path.join(save_dir, data['name'][0] + '_pure.png'))
                pbar.update(1)
    if save_mat:
        savemat(os.path.join(save_dir, 'results.mat'), {'UNetSR_sr_psnrs': sr_psnrs,
                                                        'UNetSR_sr_ssims': sr_ssims,
                                                        'names': names})
    print(f'avg psnr: sr={np.mean(sr_psnrs):5.2f}')
    print(f'avg ssim: sr={np.mean(sr_ssims):5.3f}')


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
