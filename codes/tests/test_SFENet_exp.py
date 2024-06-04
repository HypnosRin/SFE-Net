import argparse
import os

import torch
from PIL import Image
from scipy.io import savemat
from torchvision import transforms
from tqdm import tqdm

from models import get_model
from utils.universal_util import read_yaml, normalization, overlap, scan_pos, concat

import numpy as np

def test(opt):
    # pass parameter
    lr_path = opt['testing']['lr_path']
    save_img = opt['testing']['save_img']
    save_mat = opt['testing']['save_mat']
    save_dir = opt['testing']['save_dir']

    U_input = 132
    U_output = 33
    U_padding = 10
    F_scale = opt['F_Model']['network']['scale']

    # mkdir
    if (save_img['sr'] or save_img['kernel'] or save_img['sr_kernel'] or save_mat) and not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # set up model
    U_model = get_model(opt['U_Model'])
    F_model = get_model(opt['F_Model'])

    # set up recorder
    pred_kernels = []
    names = []

    # start testing
    with tqdm(desc=f'testing', total=len(os.listdir(lr_path)), unit='img') as pbar:
        with torch.no_grad():
            for name in os.listdir(lr_path):
                if '-GT' in name:
                    continue
                # img = transforms.ToTensor()(Image.open(os.path.join(lr_path, name))).float().unsqueeze(0) / 65535.0
                img = normalization(transforms.ToTensor()(Image.open(os.path.join(lr_path, name))).float().unsqueeze(0),
                                    v_min=0.0, v_max=0.4)

                # padding
                _,_,origin_x,origin_y = img.shape
                img = torch.nn.functional.pad(img,(U_padding,U_padding,U_padding,U_padding),'reflect')
                img_norm = normalization(img, 0.0, 1.0)

                # split
                positions = scan_pos(img.shape[-2], img.shape[-1], U_input, U_input, s=2)
                sr = torch.zeros(size=(1, 1, img.shape[-2] * F_scale, img.shape[-1] * F_scale))
                rt = torch.zeros(size=(1, 1, img.shape[-2] * F_scale, img.shape[-1] * F_scale))

                # mask
                pos = torch.arange(-U_input * F_scale / 2, U_input * F_scale / 2, 1)
                Y, X = torch.meshgrid(pos, pos)
                mask = U_input * F_scale - abs(X) - abs(Y) + 1
                mask[0:U_padding * F_scale,:] = 0
                mask[U_input * F_scale - U_padding * F_scale:U_input * F_scale,:] = 0
                mask[:,0:U_padding * F_scale] = 0
                mask[:,U_input * F_scale - U_padding * F_scale:U_input * F_scale] = 0
                mask /= torch.sum(mask)

                pred_kernel = []
                for pos in positions:
                    patch = img[..., pos[0]:pos[0] + U_input, pos[1]:pos[1] + U_input]
                    assert patch.shape == (1,1,132,132)
                    patch_norm = img_norm[..., pos[0]:pos[0] + U_input, pos[1]:pos[1] + U_input]
                    U_model.feed_data({'lr': patch_norm})
                    U_model.test()
                    pred_kernel.append(U_model.pred_kernel.cpu())
                    F_model.feed_data({'lr': patch, 'kernel': U_model.pred_kernel.squeeze(0)})
                    F_model.test()
                    sr[..., pos[0] * F_scale:(pos[0] + U_input) * F_scale, pos[1] * F_scale:(pos[1] + U_input) * F_scale] += F_model.sr.cpu() * mask
                    rt[..., pos[0] * F_scale:(pos[0] + U_input) * F_scale, pos[1] * F_scale:(pos[1] + U_input) * F_scale] += mask
                sr /= rt
                # remove padding
                sr = sr[..., U_padding*F_scale:(U_padding+origin_x)*F_scale, U_padding*F_scale:(U_padding+origin_y)*F_scale]

                # TODO 1 save all pred kernels 2 save averaged pred kernels

                # pred_kernels.append(np.mean(np.array([p.detach().cpu().squeeze().numpy() for p in pred_kernel]),axis=0))
                pred_kernel = concat(pred_kernel, int(len(positions) ** 0.5), int(len(positions) ** 0.5))
                pred_kernels.append(pred_kernel.detach().cpu().numpy())

                names.append(name)

                result_sr = normalization(sr).squeeze(0).squeeze(0)
                result_kernel = normalization(pred_kernel).squeeze(0).squeeze(0)
                result_all = overlap(result_kernel, result_sr, (0, 0))
                result_sr = transforms.ToPILImage()((result_sr * 65535).to(torch.int32))
                result_kernel = transforms.ToPILImage()((result_kernel * 65535).to(torch.int32))
                result_all = transforms.ToPILImage()((result_all * 65535).to(torch.int32))

                if save_img['sr']:
                    result_sr.save(os.path.join(save_dir, name))
                if save_img['kernel']:
                    result_kernel.save(os.path.join(save_dir, name.replace('.png', '_k.png')))
                if save_img['sr_kernel']:
                    result_all.save(os.path.join(save_dir, name.replace('.png', '_sr-k.png')))
                pbar.update(1)


    if save_mat:
        savemat(os.path.join(save_dir, 'results.mat'), {'UNetBased_pred_kernels': pred_kernels,
                                                        'names': names})


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
