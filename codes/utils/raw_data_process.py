import os
import shutil
import multipagetiff as mtif
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from utils.universal_util import scan_pos


def add_1to9():
    root = r'F:\DAO_WR\20230221_SUM_ki_CCPs_Clathrin'
    dirs = os.listdir(root)
    name = 'roi1_seq1_TIRF-SIM488_GreenCh.tif'
    for dir in dirs:
        if os.path.exists(os.path.join(root, dir, name)):
            s = mtif.read_stack(os.path.join(root, dir, name))
            s = np.array(s)
            img = s[0, :, :] * 0
            img = np.asarray(img, dtype=np.int32)
            for i in range(9):
                img += s[i, :, :]
            assert np.min(img) >= 0.0
            if np.max(img) > 65535:
                img = np.asarray(img, dtype=float)
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                img *= 65535
                img = np.asarray(img, dtype=np.int32)
            img = transforms.ToPILImage()(torch.from_numpy(img))
            img.save(os.path.join(root, dir, '1-9.png'))


def extract_recons():
    root = r'F:\DAO_WR\20230221_SUM_ki_CCPs_Clathrin'
    dirs = os.listdir(root)
    name = 'roi1_seq1_TIRF-SIM488_GreenCh_SIrecon.tif'
    for dir in dirs:
        if os.path.exists(os.path.join(root, dir, name)):
            s = mtif.read_stack(os.path.join(root, dir, name))
            s = np.array(s)
            img = s[0, :, :]
            img = np.asarray(img, dtype=float)
            if np.min(img) < 0:
                img -= np.min(img)
            if np.max(img) > 65535:
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                img *= 65535
            img = np.asarray(img, dtype=np.int32)
            img = transforms.ToPILImage()(torch.from_numpy(img))
            img.save(os.path.join(root, dir, 'recons.png'))

def extract_wf_ccps():
    root = r'/media/li-lab/1c1d7fee-cb9a-46f4-87fa-98f6216d5d0e/chy/KernelEstimate/PsfPred/data/exp_timeline_test_version/CCPs/WF/'
    save_root = r'/media/li-lab/1c1d7fee-cb9a-46f4-87fa-98f6216d5d0e/chy/KernelEstimate/PsfPred/data/exp_timeline_test_version/CCPs/'

    names = os.listdir(root)
    for name in names:
        save_path = os.path.join(save_root,os.path.splitext(name)[0])
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        else:
            continue
        s = mtif.read_stack(os.path.join(root, name))
        s = np.array(s)
        z,x,y = s.shape
        for iz in range(z):
            img = s[iz, :, :]
            img = np.asarray(img, dtype=np.int32)
            assert np.min(img) >= 0.0
            if np.max(img) > 65535:
                img = np.asarray(img, dtype=float)
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                img *= 65535
                img = np.asarray(img, dtype=np.int32)
            img = transforms.ToPILImage()(torch.from_numpy(img))
            img.save(os.path.join(save_path, str(iz).zfill(3)+'.png'))

def extract_wf_mts():
    root = r'/media/li-lab/1c1d7fee-cb9a-46f4-87fa-98f6216d5d0e/chy/KernelEstimate/PsfPred/data/exp_timeline_test_version/MTs_HighNA_GI-SIM/WF/'
    save_root = r'/media/li-lab/1c1d7fee-cb9a-46f4-87fa-98f6216d5d0e/chy/KernelEstimate/PsfPred/data/exp_timeline_test_version/MTs_HighNA_GI-SIM/'

    names = os.listdir(root)
    for name in names:
        save_path = os.path.join(save_root,os.path.splitext(name)[0])
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        else:
            continue
        s = mtif.read_stack(os.path.join(root, name))
        s = np.array(s)
        z,x,y = s.shape
        for iz in range(z):
            img = s[iz, :, :]
            img = np.asarray(img, dtype=np.int32)
            assert np.min(img) >= 0.0
            if np.max(img) > 65535:
                img = np.asarray(img, dtype=float)
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                img *= 65535
                img = np.asarray(img, dtype=np.int32)
            img = transforms.ToPILImage()(torch.from_numpy(img))
            img.save(os.path.join(save_path, str(iz).zfill(3)+'.png'))

def extract_wf_er():
    root = r'/mnt/data1/chy/KernelEstimate/PsfPred/data/test_data/ER/WF'
    save_root = r'/mnt/data1/chy/KernelEstimate/PsfPred/data/test_data/ER'

    names = os.listdir(root)
    for name in names:
        save_path = os.path.join(save_root,os.path.splitext(name)[0])
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        else:
            continue
        s = mtif.read_stack(os.path.join(root, name))
        s = np.array(s)
        z,x,y = s.shape
        for iz in range(z):
            img = s[iz, :, :]
            img = np.asarray(img, dtype=np.int32)
            assert np.min(img) >= 0.0
            if np.max(img) > 65535:
                img = np.asarray(img, dtype=float)
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                img *= 65535
                img = np.asarray(img, dtype=np.int32)
            img = transforms.ToPILImage()(torch.from_numpy(img))
            img.save(os.path.join(save_path, str(iz).zfill(3)+'.png'))

def copy_rename():
    root = r'F:\DAO_WR\20230221_SUM_ki_CCPs_Clathrin'
    dirs = os.listdir(root)
    for dir in dirs:
        if os.path.exists(os.path.join(root, dir, '1-9.png')):
            shutil.copy(os.path.join(root, dir, '1-9.png'), os.path.join(r'C:\Mine\PsfPred\data\exp-raw', f'{dir}.png'))
        if os.path.exists(os.path.join(root, dir, 'recons.png')):
            shutil.copy(os.path.join(root, dir, 'recons.png'), os.path.join(r'C:\Mine\PsfPred\data\exp-raw', f'{dir}-GT.png'))


def crop(h=132, w=132):
    root = r'../../data/exp-data'
    to = r'../../data/exp-crop'
    dirs = os.listdir(root)
    for dir in dirs:
        names = os.listdir(os.path.join(root, dir))
        for name in names:
            img = Image.open(os.path.join(root, dir, name))
            H, W = img.height, img.width
            postions = scan_pos(H, W, h, w)
            for i, pos in enumerate(postions):
                cropped_img = img.crop(box=(pos[1], pos[0], pos[1] + w, pos[0] + h))
                cropped_img.save(os.path.join(to, dir, name.replace('.png', f'({str(i + 1).rjust(2, "0")}).png')))


if __name__ == '__main__':
    # extract_wf_ccps()
    # extract_wf_mts()
    extract_wf_er()
    # print(scan_pos(1024, 1024, 132, 132))
    # add_1to9()
    # extract_recons()
    # copy_rename()
    # crop()
