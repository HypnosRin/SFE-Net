import torch
import torch.nn as nn


class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv_ReLU = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU())

    def forward(self, x):
        return self.conv_ReLU(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.down = nn.Sequential(nn.MaxPool2d(2),
                                  nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2,
                                            bias=True),
                                  nn.ReLU())

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)
        self.conv_relu = ConvReLU(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        return self.conv_relu(torch.cat([x2, x1], dim=1))


class UpS(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpS, self).__init__()
        self.upconv = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2),
                                    ConvReLU(in_channels, out_channels))

    def forward(self, x):
        return self.upconv(x)


class UNetSR2(nn.Module):
    def __init__(self, opt):
        super().__init__()
        img_channels = opt['img_channels']
        self.max = opt['max']
        self.min = opt['min']
        self.inconv = ConvReLU(img_channels, 64, kernel_size=3)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 32)
        self.up_s1 = UpS(64, 32)
        self.up5 = Up(64, 32)
        self.outconv = nn.Conv2d(32, img_channels, 1)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x0 = self.up_s1(x1)
        x = self.up5(x, x0)
        x = self.outconv(x)
        return torch.clamp(x, min=self.min, max=self.max)


def main():
    net = UNetSR2(opt={'img_channels': 1,
                       'max': 1.0,
                       'min': 0.0})
    print(sum(map(lambda p: p.numel(), net.parameters())))
    x = torch.rand((5, 1, 132, 132))
    net.eval()
    with torch.no_grad():
        print(net(x).shape)


if __name__ == '__main__':
    main()
