import torch.nn as nn
import torch
from torch.fft import fftn
import torch.nn.functional as F


# ------------------------------------
# Similar to the Channel Attention in SE-Net But learned in Fourier domain
# Math:
# phi(x) = GlobalPooling( ReLU( Conv( [abs(fftshift(fft2(x)))]^gamma )))
# FCALayer(x) = x * Conv_up(Conv_down(phi(x)))
# version = pytorch1.7.1
# disabled up-sample in SIM post-process tasks
# ------------------------------------

def fftshift2d(img):
    _, _, h, w = img.shape
    output = torch.cat([torch.cat([img[:, :, h // 2:, w // 2:], img[:, :, :h // 2, w // 2:]], dim=2),
                        torch.cat([img[:, :, h // 2:, :w // 2], img[:, :, :h // 2, :w // 2]], dim=2)], dim=3)
    return output


class FCALayer(nn.Module):  # 1.25
    def __init__(self, features=64, reduction=16):
        super(FCALayer, self).__init__()
        self.conv_layers_1 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=True),
            nn.ReLU())
        self.conv_layers_2 = nn.AdaptiveAvgPool2d(1)
        self.conv_layers_3 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features // reduction, kernel_size=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=features // reduction, out_channels=features, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid())

    def forward(self, x, gamma=0.8):
        x_ = fftshift2d(torch.pow(torch.abs(fftn(x, dim=[-1, -2])) + 1e-8, gamma))
        y = self.conv_layers_1(x_)
        z = self.conv_layers_2(y)
        att = self.conv_layers_3(z)
        return att * x


# ------------------------------------
# Insert channel attention (learned in Fourier domain) into ResBlocks
# Math:
# FCAB(x) = x + FCALayer( GeLU( Conv( GeLU( Conv(x)))))
# ------------------------------------
class FCAB(nn.Module):  # 3.25
    def __init__(self, features=64):
        super(FCAB, self).__init__()
        m = [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=True), nn.GELU(),
             nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=True), nn.GELU(),
             FCALayer(features=features, reduction=16)]
        self.conv_layers = nn.Sequential(*m)

    def forward(self, x):
        res = self.conv_layers(x)
        res += x
        return res


class SFT_FCAB(nn.Module):  # 3.25
    def __init__(self, nf=64, para=65):
        super(SFT_FCAB, self).__init__()
        self.mul_conv1 = nn.Conv2d(para + nf, 32, kernel_size=3, stride=1, padding=1)
        self.mul_leaky = nn.LeakyReLU(0.2)
        self.mul_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)

        self.add_conv1 = nn.Conv2d(para + nf, 32, kernel_size=3, stride=1, padding=1)
        self.add_leaky = nn.LeakyReLU(0.2)
        self.add_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)

        self.fac = FCALayer(features=nf, reduction=16)

    def forward(self, x, para_maps):
        cat_input = torch.cat((x, para_maps), dim=1)
        mul = torch.sigmoid(self.mul_conv2(self.mul_leaky(self.mul_conv1(cat_input))))
        add = self.add_conv2(self.add_leaky(self.add_conv1(cat_input)))

        res = self.fac(x * mul + add)

        res += x
        return res


# IKC: SFTMD model
class SFT_Layer(nn.Module):
    def __init__(self, nf=64, para=65):
        super(SFT_Layer, self).__init__()
        self.mul_conv1 = nn.Conv2d(para + nf, 32, kernel_size=3, stride=1, padding=1)
        self.mul_leaky = nn.LeakyReLU(0.2)
        self.mul_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)

        self.add_conv1 = nn.Conv2d(para + nf, 32, kernel_size=3, stride=1, padding=1)
        self.add_leaky = nn.LeakyReLU(0.2)
        self.add_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)

    def forward(self, feature_maps, para_maps):
        cat_input = torch.cat((feature_maps, para_maps), dim=1)
        mul = torch.sigmoid(self.mul_conv2(self.mul_leaky(self.mul_conv1(cat_input))))
        add = self.add_conv2(self.add_leaky(self.add_conv1(cat_input)))
        return feature_maps * mul + add


# ------------------------------------
# The Blocks
# Math:
# ResidualGroup(x) = x + FCAB( FCAB( FCAB( FCAB( x ))))
# ------------------------------------
class ResidualGroup(nn.Module):  # 13
    def __init__(self, features=64, n_RCAB=4):
        super(ResidualGroup, self).__init__()
        m = [FCAB(features=features) for _ in range(n_RCAB)]
        self.layers = nn.Sequential(*m)

    def forward(self, x):
        res = self.layers(x)
        res += x
        return res


class SFT_ResidualGroup(nn.Module):  # 13
    def __init__(self, features=64, n_RCAB=4, para=65):
        super(SFT_ResidualGroup, self).__init__()
        self.n_RCAB = n_RCAB
        for i in range(n_RCAB):
            self.add_module('sft' + str(i + 1), SFT_Layer(nf=features, para=para))
            self.add_module('leaky' + str(i + 1), nn.LeakyReLU(0.2))
            self.add_module('fcab' + str(i + 1), FCAB(features=features))

    def forward(self, x, ker_code):
        res = x
        for i in range(self.n_RCAB):
            res = self.__getattr__('sft' + str(i + 1))(res, ker_code)
            res = self.__getattr__('leaky' + str(i + 1))(res)
            res = self.__getattr__('fcab' + str(i + 1))(res)
        res += x
        return res

class ResidualGroup_SFT_FCAB(nn.Module):  # 13
    def __init__(self, features=64, n_RCAB=4, para=65):
        super(ResidualGroup_SFT_FCAB, self).__init__()
        self.n_RCAB = n_RCAB
        for i in range(n_RCAB):
            self.add_module('sft_fcab' + str(i + 1), SFT_FCAB(nf=features, para=para))

    def forward(self, x, ker_code):
        res = x
        for i in range(self.n_RCAB):
            res = self.__getattr__('sft_fcab' + str(i + 1))(res, ker_code)
        res += x
        return res


# ------------------------------------
# The Network
# Math:
# DFCAN_input(x) = GeLU( Conv(x))
# DFCAN_mid(x) = RG( RG( RG( RG( DFCAN_input(x) ))))
# DFCAN_output(x) = Sigmoid( Conv( GeLU( Conv( DFCAN_mid(x) ))))
# ------------------------------------
class DFCAN(nn.Module):
    def __init__(self, opt):
        '''
        DFCAN (Qiao, N.Methods 2021)
        '''
        super(DFCAN, self).__init__()

        in_nc = opt['in_nc'] if 'in_nc' in opt else 1
        out_nc = opt['out_nc'] if 'out_nc' in opt else 1
        features = opt['features'] if 'features' in opt else 64
        n_ResGroup = opt['n_ResGroup'] if 'n_ResGroup' in opt else 4
        n_RCAB = opt['n_RCAB'] if 'n_RCAB' in opt else 4

        self.input_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_nc, out_channels=features, kernel_size=3, padding=1, bias=True),
            nn.GELU())

        m = [ResidualGroup(features=features, n_RCAB=n_RCAB) for _ in range(n_ResGroup)]  # 52
        self.mid_layers = nn.Sequential(*m)

        self.output_layers = nn.Sequential(
            # output upfactor 2
            nn.Conv2d(in_channels=features, out_channels=features * 4, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.PixelShuffle(2),

            nn.Conv2d(in_channels=features, out_channels=out_nc, kernel_size=3, padding=1, bias=True))

    def forward(self, x):
        x_ = self.input_layers(x)
        x__ = self.mid_layers(x_)
        out = self.output_layers(x__)
        return out


# ------------------------------------
# The Network
# Math:
# DFCAN_input(x) = GeLU( Conv(x))
# DFCAN_mid(x) = RG( RG( RG( RG( DFCAN_input(x) ))))
# DFCAN_output(x) = Sigmoid( Conv( GeLU( Conv( DFCAN_mid(x) ))))
# ------------------------------------
class SFTDFCAN_sequential(nn.Module):
    def __init__(self, opt):
        '''
        DFCAN (Qiao, N.Methods 2021)
        '''
        super(SFTDFCAN_sequential, self).__init__()
        in_nc = opt['in_nc'] if 'in_nc' in opt else 1
        out_nc = opt['out_nc'] if 'out_nc' in opt else 1
        features = opt['features'] if 'features' in opt else 64
        n_ResGroup = opt['n_ResGroup'] if 'n_ResGroup' in opt else 4
        n_RCAB = opt['n_RCAB'] if 'n_RCAB' in opt else 4
        input_para = opt['input_para'] if 'input_para' in opt else 65

        self.input_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_nc, out_channels=features, kernel_size=3, padding=1, bias=True),
            nn.GELU())

        self.n_ResGroup = n_ResGroup
        for i in range(n_ResGroup):
            self.add_module('SFT-residual' + str(i + 1), SFT_ResidualGroup(features=features, n_RCAB=n_RCAB, para=input_para))

        self.output_layers = nn.Sequential(
            # output upfactor 2
            nn.Conv2d(in_channels=features, out_channels=features * 4, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.PixelShuffle(2),

            nn.Conv2d(in_channels=features, out_channels=out_nc, kernel_size=3, padding=1, bias=True))

    def forward(self, x, ker_code):
        """x of (B, C, H, W), ker_code of (B, h)"""
        B, C, H, W = x.size()  # I_LR batch
        B_h, C_h = ker_code.size()  # Batch, Len=10
        ker_code_exp = ker_code.view((B_h, C_h, 1, 1)).expand((B_h, C_h, H, W))  # kernel_map stretch

        x_ = self.input_layers(x)
        for i in range(self.n_ResGroup):
            x_ = self.__getattr__('SFT-residual' + str(i + 1))(x_, ker_code_exp)

        out = self.output_layers(x_)
        return out

class SFTDFCAN(nn.Module):
    def __init__(self, opt):
        super(SFTDFCAN, self).__init__()
        in_nc = opt['in_nc'] if 'in_nc' in opt else 1
        out_nc = opt['out_nc'] if 'out_nc' in opt else 1
        features = opt['features'] if 'features' in opt else 64
        n_ResGroup = opt['n_ResGroup'] if 'n_ResGroup' in opt else 4
        n_RCAB = opt['n_RCAB'] if 'n_RCAB' in opt else 4
        input_para = opt['input_para'] if 'input_para' in opt else 65

        self.input_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_nc, out_channels=features, kernel_size=3, padding=1, bias=True),
            nn.GELU())

        self.n_ResGroup = n_ResGroup
        for i in range(n_ResGroup):
            self.add_module('SFT-residual' + str(i + 1), ResidualGroup_SFT_FCAB(features=features, n_RCAB=n_RCAB, para=input_para))

        self.output_layers = nn.Sequential(
            # output upfactor 2
            nn.Conv2d(in_channels=features, out_channels=features * 4, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.PixelShuffle(2),

            nn.Conv2d(in_channels=features, out_channels=out_nc, kernel_size=3, padding=1, bias=True))

    def forward(self, x, ker_code):
        """x of (B, C, H, W), ker_code of (B, h)"""
        B, C, H, W = x.size()  # I_LR batch
        B_h, C_h = ker_code.size()  # Batch, Len=10
        ker_code_exp = ker_code.view((B_h, C_h, 1, 1)).expand((B_h, C_h, H, W))  # kernel_map stretch

        x_ = self.input_layers(x)
        for i in range(self.n_ResGroup):
            x_ = self.__getattr__('SFT-residual' + str(i + 1))(x_, ker_code_exp)

        out = self.output_layers(x_)
        return out
