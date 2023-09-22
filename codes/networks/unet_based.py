import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


# basic component


class CA(nn.Module):
    """channel attention, not change (N, C, H, W)"""

    def __init__(self, num_features=64, reduction=16):
        super().__init__()
        self.module = nn.Sequential(nn.AdaptiveAvgPool2d(1),  # (N, C, H, W) => (N, C, 1, 1)
                                    nn.Conv2d(num_features, num_features // reduction, kernel_size=1, padding=0),
                                    nn.ReLU(),
                                    nn.Conv2d(num_features // reduction, num_features, kernel_size=1, padding=0),
                                    nn.Sigmoid())

    def forward(self, x):
        return x * self.module(x)


class RCAB(nn.Module):
    """residual channel attention block, not change (N, C, H, W)"""

    def __init__(self, num_features=64, reduction=16):
        super().__init__()
        self.module = nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                    CA(num_features, reduction))

    def forward(self, x):
        return x + self.module(x)


class RG(nn.Module):
    """residual group, not change (N, C, H, W)"""

    def __init__(self, num_features=64, reduction=16, num_rcab=20):
        super().__init__()
        self.module = nn.Sequential(*[RCAB(num_features, reduction) for _ in range(num_rcab)])
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))

    def forward(self, x):
        return x + self.module(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        return self.double_conv(x)


class ResDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.channel_adjust_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        return self.double_conv(x) + self.channel_adjust_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2),
                                          DoubleConv(in_channels, out_channels, mid_channels, kernel_size))

    def forward(self, x):
        return self.maxpool_conv(x)


class ResDown(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2),
                                          ResDoubleConv(in_channels, out_channels, mid_channels, kernel_size))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        self.up = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels // 4, in_channels // 2, kernel_size, padding=kernel_size // 2, bias=True))
        self.conv = DoubleConv(in_channels, out_channels, mid_channels, kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ResUp(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        self.up = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels // 4, in_channels // 2, kernel_size, padding=kernel_size // 2, bias=True))
        self.conv = ResDoubleConv(in_channels, out_channels, mid_channels, kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class PixelStack(nn.Module):
    """(N, C, H, W) => (N, C* scale**2, H/scale, W/scale)"""
    def __init__(self, scale):
        super().__init__()
        assert scale >= 1 and isinstance(scale, int), f'scale factor must be int, but got {self.scale}'
        self.scale = scale

    def squeeze2d(self, x):
        if self.scale == 1:
            return x
        B, C, H, W = x.shape
        assert H % self.scale == 0 and W % self.scale == 0, f'{(H, W)} should be an integral multiple of scale'
        out = x.view(B, C, H // self.scale, self.scale, W // self.scale, self.scale)
        return out.permute(0, 1, 3, 5, 2, 4).contiguous().view(B, C * self.scale ** 2, H // self.scale, W // self.scale)

    def forward(self, x):
        return self.squeeze2d(x)


class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv_ReLU = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU())

    def forward(self, x):
        return self.conv_ReLU(x)


# assembled network


class RCAN(nn.Module):
    """residual channel attention network, (N, C, H, W) => (N, C, H * scale, W * scale)"""

    def __init__(self, opt):
        super().__init__()
        in_channels = opt['in_channels']
        scale = opt['scale']
        num_rg = opt['num_rg'] if opt['num_rg'] is not None else 10
        num_features = opt['num_features'] if opt['num_features'] is not None else 64
        reduction = opt['reduction'] if opt['reduction'] is not None else 16
        num_rcab = opt['num_rcab'] if opt['num_rcab'] is not None else 20
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.rgs = nn.Sequential(*[RG(num_features, reduction, num_rcab) for _ in range(num_rg)])
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.upscale = nn.Sequential(nn.Conv2d(num_features, num_features * (scale ** 2), kernel_size=3, padding=1),
                                     nn.PixelShuffle(scale))
        self.conv3 = nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        residual = x
        x = self.conv2(self.rgs(x))
        x += residual
        x = self.conv3(self.upscale(x))
        return x


class RCANEncoder(nn.Module):
    """RCAN encoder, (N, in_channels, H, W) => (N, num_features, H, W)"""

    def __init__(self, opt):
        super().__init__()
        in_channels = opt['in_channels']
        num_rg = opt['num_rg'] if opt['num_rg'] is not None else 10
        num_features = opt['num_features'] if opt['num_features'] is not None else 64
        reduction = opt['reduction'] if opt['reduction'] is not None else 16
        num_rcab = opt['num_rcab'] if opt['num_rcab'] is not None else 20
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.rgs = nn.Sequential(*[RG(num_features, reduction, num_rcab) for _ in range(num_rg)])
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        residual = x
        x = self.conv2(self.rgs(x))
        x += residual
        x = self.conv3(x)
        return x


class UNetBased(nn.Module):
    def __init__(self, opt):
        super().__init__()
        in_channels = opt['in_channels']
        out_channels = opt['out_channels']

        self.in_conv = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.down5 = Down(1024, 2048)
        self.up5 = Down(2048, 1024)
        self.up4 = Up(1024, 512)
        self.up3 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up1 = Up(128, 64)

        self.stack = PixelStack(2)
        self.out_layer1 = ConvReLU(256, 64)
        self.out_layer2 = ConvReLU(256, 64)
        self.out_layer3 = ConvReLU(256, 64)
        self.out_layer4 = ConvReLU(64, 8)
        self.out_layer5 = ConvReLU(8, out_channels)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        y = self.up5(x6, x5)
        y = self.up4(y, x4)
        y = self.up3(y, x3)
        y = self.up2(y, x2)
        y = self.up1(y, x1)
        y = self.out_layer1(self.stack(y))
        y = self.out_layer2(self.stack(y))
        y = self.out_layer3(self.stack(y))
        return self.out_layer5(self.out_layer4(y))


class ResUNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        in_channels = opt['in_channels']
        out_channels = opt['out_channels']
        self.num_pixel_stack_layer = opt['num_pixel_stack_layer']
        self.num_down_up = opt['num_down_up']
        assert self.num_down_up in (4, 5, 6)

        self.in_conv = ResDoubleConv(in_channels, 64)
        self.down1 = ResDown(64, 128)
        self.down2 = ResDown(128, 256)
        self.down3 = ResDown(256, 512)
        self.down4 = ResDown(512, 1024)
        self.up4 = ResUp(1024, 512)
        self.up3 = ResUp(512, 256)
        self.up2 = ResUp(256, 128)
        self.up1 = ResUp(128, 64)
        if self.num_down_up >= 5:
            self.down5 = ResDown(1024, 2048)
            self.up5 = ResUp(2048, 1024)
            if self.num_down_up >= 6:
                self.down6 = ResDown(2048, 4096)
                self.up6 = ResUp(4096, 2048)
        if self.num_pixel_stack_layer >= 1:
            self.down_sample = nn.Sequential(
                *[nn.Sequential(PixelStack(2), ConvReLU(256, 64)) for _ in range(self.num_pixel_stack_layer)])
        self.out_layer1 = ConvReLU(64, 8)
        self.out_layer2 = ConvReLU(8, out_channels)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.num_down_up >= 5:
            x6 = self.down5(x5)
            if self.num_down_up >= 6:
                x7 = self.down6(x6)
                out = self.up6(x7, x6)
                out = self.up5(out, x5)
                out = self.up4(out, x4)
            else:
                out = self.up5(x6, x5)
                out = self.up4(out, x4)
        else:
            out = self.up4(x5, x4)
        out = self.up3(out, x3)
        out = self.up2(out, x2)
        out = self.up1(out, x1)
        if self.num_pixel_stack_layer >= 1:
            out = self.down_sample(out)
        return self.out_layer2(self.out_layer1(out))


class FFTResUNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        in_channels = opt['in_channels']
        encoder_channels = opt['encoder_channels']
        _ = opt['out_channels'], opt['num_pixel_stack_layer'], opt['num_down_up']
        assert opt['num_down_up'] in (4, 5, 6)
        assert encoder_channels % 2 == 0
        self.in_conv_relu1 = ConvReLU(in_channels, encoder_channels // 2)
        self.in_conv_relu2 = ConvReLU(in_channels, encoder_channels // 2)
        res_unet_opt = deepcopy(opt)
        res_unet_opt['in_channels'] = encoder_channels
        self.res_unet = ResUNet(res_unet_opt)

    def forward(self, x):
        x1 = self.in_conv_relu1(x)
        x2 = self.in_conv_relu2(x)
        x2 = torch.fft.fft2(x2)
        x2 = torch.abs(x2)
        x2 = torch.log10(1 + x2)
        ux = torch.cat([x1, x2], dim=-3)
        out = self.res_unet(ux)
        return out


class FFTRCANResUNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.fft = opt['fft']
        self.fft_shift = opt['fft_shift']
        self.fft_forward = opt['fft_forward']
        self.fft_brunch = opt['fft_brunch']
        encoder_channels = opt['encoder_channels']
        _ = opt['in_channels'], opt['out_channels'], opt['num_rg'], opt['num_rcab'], opt['reduction'], \
            opt['num_down_up'], opt['num_pixel_stack_layer']
        assert opt['num_down_up'] in (4, 5, 6)
        assert encoder_channels % 2 == 0

        rcan_opt = deepcopy(opt)

        self.in_rcan1 = RCANEncoder(rcan_opt)
        self.in_rcan2 = RCANEncoder(rcan_opt)

        res_unet_opt = deepcopy(opt)
        if self.fft_brunch:
            res_unet_opt['in_channels'] = encoder_channels
        else:
            res_unet_opt['in_channels'] = encoder_channels // 2

        self.res_unet = ResUNet(res_unet_opt)

    ## version: fft brunch & fft forward
    # def forward(self, x):
    #     x1 = self.in_rcan1(x)
    #     if self.fft_brunch:
    #         if self.fft_forward:
    #             x2 = torch.fft.fft2(x)
    #             x2 = torch.abs(x2)
    #             x2 = torch.log10(1 + x2)
    #             x2 = self.in_rcan2(x2)
    #         else:
    #             x2 = self.in_rcan2(x)
    #             x2 = torch.fft.fft2(x2)
    #             x2 = torch.abs(x2)
    #             x2 = torch.log10(1 + x2)
    #         ux = torch.cat([x1, x2], dim=-3)
    #     else:
    #         ux = x1
    #     out = self.res_unet(ux)
    #     return out

    ## version: fft brunch & fft forward & fft shift
    # def forward(self, x):
    #     x1 = self.in_rcan1(x)
    #     if self.fft_brunch:
    #         if self.fft_forward:
    #             x2 = torch.fft.fft2(x)
    #             if self.fft_shift:
    #                 x2 = torch.fft.fftshift(x2)
    #             x2 = torch.abs(x2)
    #             x2 = torch.log10(1 + x2)
    #             x2 = self.in_rcan2(x2)
    #         else:
    #             x2 = self.in_rcan2(x)
    #             x2 = torch.fft.fft2(x2)
    #             if self.fft_shift:
    #                 x2 = torch.fft.fftshift(x2)
    #             x2 = torch.abs(x2)
    #             x2 = torch.log10(1 + x2)
    #         ux = torch.cat([x1, x2], dim=-3)
    #     else:
    #         ux = x1
    #     out = self.res_unet(ux)
    #     return out

    ## version: fft brunch & fft forward & fft shift & cepstral
    def forward(self, x):
        x1 = self.in_rcan1(x)
        if self.fft_brunch:
            if self.fft_forward:
                x2 = x
            else:
                x2 = self.in_rcan2(x)

            if self.fft:
                x2 = torch.fft.fft2(x2)
                x2 = torch.abs(x2)
                x2 = torch.log10(1 + x2)
                if self.fft_shift:
                    x2 = torch.fft.fftshift(x2)

            if self.fft_forward:
                x2 = self.in_rcan2(x2)

            ux = torch.cat([x1, x2], dim=-3)
        else:
            ux = x1
        out = self.res_unet(ux)
        return out


def main():
    net = FFTRCANResUNet(opt={'in_channels': 1,
                              'encoder_channels': 128,
                              'num_rg': 2,
                              'num_rcab': 4,
                              'reduction': 16,
                              'num_down_up': 4,
                              'num_pixel_stack_layer': 3,
                              'out_channels': 1})
    print(sum(map(lambda p: p.numel(), net.parameters())))
    x = torch.rand((5, 1, 264, 264))
    net.eval()
    with torch.no_grad():
        print(net(x).shape)


if __name__ == '__main__':
    main()
