import argparse
import os

import torch
from PIL import Image
from scipy.io import savemat
from torchvision import transforms
from tqdm import tqdm

from models import get_model
from utils.universal_util import read_yaml, normalization, PCA_Decoder


def test(opt):
    # pass parameter
    lr_path = opt['testing']['lr_path']
    correct_step = opt['testing']['correct_step']
    save_img = opt['testing']['save_img']
    save_mat = opt['testing']['save_mat']
    save_dir = opt['testing']['save_dir']

    # mkdir
    if (save_img or save_mat) and not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # set up model
    F_model = get_model(opt['F_model'])
    P_model = get_model(opt['P_model'])
    C_model = get_model(opt['C_model'])
    pca_decoder = PCA_Decoder(weight=F_model.pca_encoder.weight, mean=F_model.pca_encoder.mean)

    # set up recorder
    pred_kernels = []
    names = []

    # start testing
    with tqdm(desc=f'Testing', total=len(os.listdir(lr_path)), unit='img') as pbar:
        with torch.no_grad():
            for name in os.listdir(lr_path):
                img = transforms.ToTensor()(Image.open(os.path.join(lr_path, name))).float().unsqueeze(0) / 65535.0
                P_model.feed_data({'lr': img, 'kernel': torch.rand((1, 33, 33))})
                P_model.test()
                kernel_code_of_sr = P_model.pred_kernel_code.detach().cpu()
                for i in range(correct_step):
                    F_model.feed_data({'hr': torch.rand(size=img.shape),
                                       'lr': img,
                                       'kernel_code': kernel_code_of_sr})
                    F_model.test()
                    sr = F_model.sr.detach().cpu()
                    C_model.feed_data({'sr': sr,
                                       'kernel_code_of_sr': kernel_code_of_sr,
                                       'gt_kernel_code': torch.rand(size=kernel_code_of_sr.shape)})
                    C_model.test()
                    if i <= correct_step - 2:
                        kernel_code_of_sr = C_model.pred_kernel_code.detach().cpu()
                    else:
                        kernel_code_of_sr = kernel_code_of_sr.to(F_model.device)
                pred_kernel = pca_decoder(kernel_code_of_sr).squeeze(0)
                pred_kernels.append(pred_kernel.detach().cpu().numpy())
                names.append(name)

                result_sr = normalization(F_model.sr).squeeze(0).squeeze(0)
                result_kernel = normalization(pred_kernel).squeeze(0).squeeze(0)
                result_sr = transforms.ToPILImage()((result_sr * 65535).to(torch.int32))
                result_kernel = transforms.ToPILImage()((result_kernel * 65535).to(torch.int32))

                if save_img:
                    result_sr.save(os.path.join(save_dir, name))
                    result_kernel.save(os.path.join(save_dir, name.replace('.png', '_k.png')))
                pbar.update(1)
    if save_mat:
        savemat(os.path.join(save_dir, 'results.mat'), {'IKC_pred_kernels': pred_kernels,
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
