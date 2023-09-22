import numpy as np
import math
import torch


def cal_sigma(sigma_x, sigma_y, radians):
    D = np.array([[sigma_x ** 2, 0], [0, sigma_y ** 2]])
    U = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), 1 * np.cos(radians)]])
    sigma = np.dot(U, np.dot(D, U.T))
    return sigma


def anisotropic_gaussian_kernel(kernel_size, sigma_matrix, tensor=False):
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((kernel_size * kernel_size, 1)),
                    yy.reshape(kernel_size * kernel_size, 1))).reshape(kernel_size, kernel_size, 2)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(xy, inverse_sigma) * xy, 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def isotropic_gaussian_kernel(kernel_size, sigma, tensor=False):
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def random_anisotropic_gaussian_kernel(sigma_min=0.2, sigma_max=4.0, scale=3, kernel_size=21, tensor=False):
    pi = np.random.random() * math.pi * 2 - math.pi
    x = np.random.random() * (sigma_max - sigma_min) + sigma_min
    y = np.clip(np.random.random() * scale * x, sigma_min, sigma_max)
    sig = cal_sigma(x, y, pi)
    k = anisotropic_gaussian_kernel(kernel_size, sig, tensor=tensor)
    return k


def random_isotropic_gaussian_kernel(sigma_min=0.2, sigma_max=4.0, kernel_size=21, tensor=False):
    x = np.random.random() * (sigma_max - sigma_min) + sigma_min
    k = isotropic_gaussian_kernel(kernel_size, x, tensor=tensor)
    return k


def stable_isotropic_gaussian_kernel(sigma=2.6, kernel_size=21, tensor=False):
    x = sigma
    k = isotropic_gaussian_kernel(kernel_size, x, tensor=tensor)
    return k


def random_gaussian_kernel(kernel_size=21, sigma_min=0.2, sigma_max=4.0, prob_isotropic=1.0, scale=3, tensor=False):
    if np.random.random() < prob_isotropic:
        return random_isotropic_gaussian_kernel(kernel_size=kernel_size, sigma_min=sigma_min, sigma_max=sigma_max,
                                                tensor=tensor)
    else:
        return random_anisotropic_gaussian_kernel(kernel_size=kernel_size, sigma_min=sigma_min, sigma_max=sigma_max,
                                                  scale=scale, tensor=tensor)


def stable_gaussian_kernel(kernel_size=21, sigma=2.6, tensor=False):
    return stable_isotropic_gaussian_kernel(sigma=sigma, kernel_size=kernel_size, tensor=tensor)


def random_batch_kernel(batch_size, kernel_size=21, sigma_min=0.2, sigma_max=4.0, prob_isotropic=1.0, scale=3,
                        tensor=True):
    batch_kernel = np.zeros((batch_size, kernel_size, kernel_size))
    for i in range(batch_size):
        batch_kernel[i] = random_gaussian_kernel(kernel_size=kernel_size, sigma_min=sigma_min, sigma_max=sigma_max,
                                                 prob_isotropic=prob_isotropic, scale=scale, tensor=False)
    return torch.FloatTensor(batch_kernel) if tensor else batch_kernel


def stable_batch_kernel(batch_size, kernel_size=21, sigma=2.6, tensor=True):
    batch_kernel = np.zeros((batch_size, kernel_size, kernel_size))
    for i in range(batch_size):
        batch_kernel[i] = stable_gaussian_kernel(kernel_size=kernel_size, sigma=sigma, tensor=False)
    return torch.FloatTensor(batch_kernel) if tensor else batch_kernel


class GaussianKernelGenerator(object):
    def __init__(self, opt):
        self.kernel_size = opt['kernel_size']
        self.sigma = opt['sigma']
        self.sigma_min = opt['sigma_min']
        self.sigma_max = opt['sigma_max']
        self.prob_isotropic = opt['prob_isotropic']
        self.scale = opt['scale']

    def __call__(self, batch_size, tensor=False, random=True):
        """
        :param batch_size:
        :param tensor: if False, return ndarray, else return Tensor
        :param random:
        :return: kernels of (batch_size, kernel_size, kernel_size)
        """
        if random:  # random kernel
            return random_batch_kernel(batch_size, kernel_size=self.kernel_size, sigma_min=self.sigma_min,
                                       sigma_max=self.sigma_max, prob_isotropic=self.prob_isotropic,
                                       scale=self.scale, tensor=tensor)
        else:  # stable kernel
            return stable_batch_kernel(batch_size, kernel_size=self.kernel_size, sigma=self.sigma, tensor=tensor)


def main():
    gen = GaussianKernelGenerator(opt={'kernel_size': 33,
                                       'sigma': 2.6,
                                       'sigma_min': 0.2,
                                       'sigma_max': 4.0,
                                       'prob_isotropic': 0.0,
                                       'scale': 2})
    k = gen(64, False)
    k = gen(64, True)
    print(k)


if __name__ == '__main__':
    main()
