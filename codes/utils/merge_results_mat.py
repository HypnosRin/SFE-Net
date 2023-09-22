
from scipy.io import savemat, loadmat
import numpy as np

dic = ['KernelGAN_pred_kernels', 'KernelGAN_kernel_psnrs', 'KernelGAN_kernel_ssims', 'KernelGAN_sr_psnrs',
       'KernelGAN_sr_ssims', 'names']

# todo config
path = "../PsfPred/results/zernike-retrain-120item/KernelGAN-5s/"
n = 3

ker_merge = loadmat(path + "results1.mat")

for idx in range(2,n+1):
       ker = loadmat(path + "results"+str(idx)+".mat")
       # ker2 = loadmat(path + "results2.mat")
       # for d in dic:
       #     ker_merge[d] = np.concatenate((ker1[d], ker2[d]), axis=0)

       ker_merge[dic[0]] = np.concatenate((ker_merge[dic[0]], ker[dic[0]]), axis=0)
       ker_merge[dic[1]] = np.concatenate((ker_merge[dic[1]], ker[dic[1]]), axis=1)
       ker_merge[dic[2]] = np.concatenate((ker_merge[dic[2]], ker[dic[2]]), axis=1)
       ker_merge[dic[3]] = np.concatenate((ker_merge[dic[3]], ker[dic[3]]), axis=1)
       ker_merge[dic[4]] = np.concatenate((ker_merge[dic[4]], ker[dic[4]]), axis=1)
       ker_merge[dic[5]] = np.concatenate((ker_merge[dic[5]], ker[dic[5]]), axis=0)

savemat(path + "results.mat", ker_merge)
