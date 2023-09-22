import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class MAConv(nn.Module):
    """
    Mutual Affine Convolution (MAConv) layer
    (B, in_channels, H, W) -> (B, out_channels, H, W)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, split=2, reduction=2):
        """
        in_channels: input channel
        out_channels: output channel
        kernel_size, stride, padding, bias: args for Conv2d
        split: number of branches
        reduction: for affine transformation module
        """
        super().__init__()
        assert split >= 2, 'Num of splits should be larger than one'

        self.num_split = split
        splits = [1 / split] * split
        self.in_split, self.in_split_rest, self.out_split = [], [], []

        for i in range(self.num_split):
            in_split = round(in_channels * splits[i]) if i < self.num_split - 1 else in_channels - sum(self.in_split)
            in_split_rest = in_channels - in_split
            out_split = round(out_channels * splits[i]) if i < self.num_split - 1 else in_channels - sum(self.out_split)

            self.in_split.append(in_split)
            self.in_split_rest.append(in_split_rest)
            self.out_split.append(out_split)

            setattr(self, f'fc{i}', nn.Sequential(*[
                nn.Conv2d(in_channels=in_split_rest, out_channels=int(in_split_rest // reduction),
                          kernel_size=1, stride=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=int(in_split_rest // reduction), out_channels=in_split * 2,
                          kernel_size=1, stride=1, padding=0, bias=True),
            ]))
            setattr(self, f'conv{i}', nn.Conv2d(in_channels=in_split, out_channels=out_split,
                                                kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))

    def forward(self, x):
        # x: (B, C, H, W) -> tuple( (B, self.in_split[0], H, W), (B, self.in_split[1], H, W), ... )
        x = torch.split(x, self.in_split, dim=1)
        output = []

        for i in range(self.num_split):
            # torch.cat(x[:i] + x[i + 1:])与x[i]在channel上互补
            scale, translation = torch.split(getattr(self, f'fc{i}')(torch.cat(x[:i] + x[i + 1:], 1)),
                                             [self.in_split[i], self.in_split[i]], dim=1)
            output.append(getattr(self, f'conv{i}')(x[i] * torch.sigmoid(scale) + translation))

        return torch.cat(output, 1)


class MABlock(nn.Module):
    """
    Residual block based on MAConv
    (B, in_channels, H, W) -> (B, out_channels, H, W)
    """

    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True,
                 split=2, reduction=2):
        super().__init__()

        self.res = nn.Sequential(*[
            MAConv(in_channels, in_channels, kernel_size, stride, padding, bias, split, reduction),
            nn.ReLU(inplace=True),
            MAConv(in_channels, out_channels, kernel_size, stride, padding, bias, split, reduction),
        ])

    def forward(self, x):
        return x + self.res(x)


class MANet(nn.Module):
    """
    Network of MANet
    (B, in_nc, H, W) -> (B, kernel_size**2, H, W)
    """

    def __init__(self, in_nc=3, kernel_size=21, nc=(128, 256), nb=1, split=2):
        super().__init__()
        self.kernel_size = kernel_size

        self.m_head = nn.Conv2d(in_channels=in_nc, out_channels=nc[0], kernel_size=3, padding=1, bias=True)
        self.m_down1 = sequential(*[MABlock(nc[0], nc[0], bias=True, split=split) for _ in range(nb)],
                                  nn.Conv2d(in_channels=nc[0], out_channels=nc[1], kernel_size=2, stride=2, padding=0,
                                            bias=True))

        self.m_body = sequential(*[MABlock(nc[1], nc[1], bias=True, split=split) for _ in range(nb)])

        self.m_up1 = sequential(nn.ConvTranspose2d(in_channels=nc[1], out_channels=nc[0],
                                                   kernel_size=2, stride=2, padding=0, bias=True),
                                *[MABlock(nc[0], nc[0], bias=True, split=split) for _ in range(nb)])
        self.m_tail = nn.Conv2d(in_channels=nc[0], out_channels=kernel_size ** 2, kernel_size=3, padding=1, bias=True)

        self.softmax = nn.Softmax(1)

    def forward(self, x):
        H, W = x.shape[-2:]
        paddingBottom = math.ceil(H / 8) * 8 - H  # such that (H + paddingBottom) % 8 == 0
        paddingRight = math.ceil(W / 8) * 8 - W  # such that (W + paddingRight) % 8 == 0
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)  # (B, in_nc, H, W) -> (B, nc[0], H, W)
        x2 = self.m_down1(x1)  # (B, nc[0], H, W) -> (B, nc[1], H//2, W//2)
        x = self.m_body(x2)  # (B, nc[1], H//2, W//2) -> (B, nc[1], H//2, W//2)
        x = self.m_up1(x + x2)  # (B, nc[1], H//2, W//2) -> (B, nc[0], H, W)
        x = self.m_tail(x + x1)  # (B, nc[0], H, W) -> (B, kernel_size**2, H, W)

        x = x[..., :H, :W]  # remove ReplicationPad part

        return self.softmax(x)


class MANet_s1(nn.Module):
    """
    stage1, train MANet 仅用于估计kernel
    (B, in_nc, H, W) -> (B, H*scale, W*scale, kernel_size, kernel_size)
    """

    def __init__(self, opt):
        super().__init__()
        self.in_nc = opt['in_nc']
        self.scale = opt['scale']
        self.kernel_size = opt['kernel_size']
        self.manet_nf = opt['manet_nf']
        self.manet_nb = opt['manet_nb']
        self.split = opt['split']
        self.kernel_estimation = MANet(in_nc=self.in_nc, kernel_size=self.kernel_size,
                                       nc=[self.manet_nf, self.manet_nf * 2], nb=self.manet_nb, split=self.split)

    def forward(self, x):
        """
        :param x: LR of (B, C, H, W), [0, 1]
        :return: kernels of shape (B, H*scale*W*scale, kernel_h, kernel_w), represent kernels of every HR pixels
        """
        kernel = self.kernel_estimation(x)
        # because of the nearest interpolation, every (scale * scale) areas of HR share the same kernel
        kernel = F.interpolate(kernel, scale_factor=self.scale, mode='nearest').flatten(2).permute(0, 2, 1)
        kernel = kernel.view(-1, kernel.size(1), self.kernel_size, self.kernel_size)
        return kernel


def main():
    net = MANet_s1(opt={'in_nc': 1,
                        'scale': 1,
                        'kernel_size': 33,
                        'manet_nf': 128,
                        'manet_nb': 1,
                        'split': 2})
    print(sum(map(lambda p: p.numel(), net.parameters())))
    x = torch.rand((5, 1, 264, 264))
    net.eval()
    with torch.no_grad():
        print(net(x).shape)


if __name__ == '__main__':
    main()
