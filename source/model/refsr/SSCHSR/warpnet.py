import torch.nn as nn
import torch

from ..qrnn3d.layer import QRNNConv3D, QRNNDeConv3D, QRNNUpsampleConv3d, BiQRNNDeConv3D, BiQRNNConv3D, ConvBlock


def conv_activation(in_ch, out_ch , kernel_size = 3, stride = 1, padding = 1, activation = 'relu', groups = 1 , init_type = 'w_init_relu'):
    if activation == 'relu':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding, groups = groups),
                nn.ReLU(inplace = True))

    elif activation == 'leaky_relu':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding, groups = groups),
                nn.LeakyReLU(negative_slope = 0.1 ,inplace = True ))

    elif activation == 'selu':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding, groups = groups),
                nn.SELU(inplace = True))

    elif activation == 'linear':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding, groups = groups))
    
class UNet(nn.Module):
    def __init__(self, block=ConvBlock,dim=32):
        super(UNet, self).__init__()

        self.dim = dim
        self.ConvBlock1 = ConvBlock(9, dim, strides=1)
        self.pool1 = nn.Conv2d(dim,dim,kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = block(dim, dim*2, strides=1)
        self.pool2 = nn.Conv2d(dim*2,dim*2,kernel_size=4, stride=2, padding=1)

        self.ConvBlock3 = block(dim*2, dim*4, strides=1)

        self.upv4 = nn.ConvTranspose2d(dim*4, dim*2, 2, stride=2)
        self.ConvBlock4 = block(dim*4, dim*2, strides=1)

        self.upv5 = nn.ConvTranspose2d(dim*2, dim, 2, stride=2)
        self.ConvBlock5 = block(dim*2, dim, strides=1)

        self.conv6 = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, hsi_rgb_sr, ref_hr, ref_warp):
        x = torch.cat([hsi_rgb_sr, ref_hr, ref_warp], dim=1)
        conv1 = self.ConvBlock1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2)

        up4 = self.upv4(conv3)
        up4 = torch.cat([up4, conv2], 1)
        conv4 = self.ConvBlock4(up4)

        up5 = self.upv5(conv4)
        up5 = torch.cat([up5, conv1], 1)
        conv5 = self.ConvBlock5(up5)

        conv6 = self.conv6(conv5)

        return conv6 + ref_warp
    
class warpnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = UNet(dim=16)

    def forward(self, hsi_rgb_sr, ref_hr, ref_warp):
        out = self.block(hsi_rgb_sr, ref_hr, ref_warp)
        return out