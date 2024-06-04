import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        struct = opt['G_structure']
        out_channels = opt['G_chan']
        scale_factor = opt['scale_factor']
        input_crop_size = opt['input_crop_size']

        # First layer - Converting RGB image to latent space
        self.first_layer = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=struct[0], bias=False)
        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            feature_block += [nn.Conv2d(out_channels, out_channels, kernel_size=struct[layer], bias=False)]
        self.feature_block = nn.Sequential(*feature_block)
        # Final layer - Down-sampling and converting back to image
        self.final_layer = nn.Conv2d(in_channels=out_channels, out_channels=1, kernel_size=struct[-1],
                                     stride=int(1 / scale_factor), bias=False)
        # Calculate number of pixels shaved in the forward pass
        self.output_size = self.forward(torch.FloatTensor(
            torch.ones([1, 1, input_crop_size, input_crop_size]))).shape[-1]
        self.forward_shave = int(input_crop_size * scale_factor) - self.output_size

    def forward(self, input_tensor):
        """(1, C, H, W) => (1, C, h, w))"""
        # Swap axis of RGB image for the network to get a "batch" of size = 3 rather the 3 channels
        input_tensor = input_tensor.transpose(0, 1)
        downscaled = self.first_layer(input_tensor)
        features = self.feature_block(downscaled)
        output = self.final_layer(features)
        return output.transpose(0, 1)


class Discriminator(nn.Module):

    def __init__(self, opt):
        super().__init__()
        img_channel = opt['img_channel']
        D_chan = opt['D_chan']
        D_kernel_size = opt['D_kernel_size']
        D_n_layers = opt['D_n_layers']
        input_crop_size = opt['input_crop_size']

        # First layer - Convolution (with no ReLU)
        self.first_layer = nn.utils.spectral_norm(nn.Conv2d(img_channel, D_chan, kernel_size=D_kernel_size, bias=True))
        feature_block = []  # Stacking layers with 1x1 kernels
        for _ in range(1, D_n_layers - 1):
            feature_block += [nn.utils.spectral_norm(nn.Conv2d(D_chan, D_chan, kernel_size=1, bias=True)),
                              nn.BatchNorm2d(D_chan),
                              nn.ReLU(True)]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(D_chan, 1, kernel_size=1, bias=True)),
                                         nn.Sigmoid())
        # Calculate number of pixels shaved in the forward pass
        self.forward_shave = input_crop_size - self.forward(torch.FloatTensor(
            torch.ones([1, img_channel, input_crop_size, input_crop_size]))).shape[-1]

    def forward(self, input_tensor):
        receptive_extraction = self.first_layer(input_tensor)
        features = self.feature_block(receptive_extraction)
        return self.final_layer(features)


def main():
    net_G = Generator(opt={'G_structure': (7, 7, 5, 5, 5, 3, 3, 3, 3, 1, 1, 1),
                           'G_chan': 64,
                           'scale_factor': 1.0,
                           'input_crop_size': 64})
    print(sum(map(lambda p: p.numel(), net_G.parameters())))
    x = torch.rand((1, 3, 64, 64))
    net_G.eval()
    with torch.no_grad():
        print(net_G(x).shape)

    net_D = Discriminator(opt={'img_channel': 3,
                               'D_chan': 64,
                               'D_kernel_size': 7,
                               'D_n_layers': 7,
                               'input_crop_size': 64})
    print(sum(map(lambda p: p.numel(), net_D.parameters())))
    x = torch.rand((1, 3, net_G.output_size, net_G.output_size))
    net_D.eval()
    with torch.no_grad():
        print(net_D(x).shape)


if __name__ == '__main__':
    main()
