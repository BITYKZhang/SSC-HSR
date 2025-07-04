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

# 三级特征
class HSIEncoder(nn.Module):
    def __init__(self, in_channels, channels, block=ConvBlock, bn=False, act='tanh'):
        super(HSIEncoder, self).__init__()
        print("HSIEncoder!!!")
        # Encoder        
        self.ConvBlock1 = block(in_channels, channels, strides=1)
        self.pool1 = nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2)
        
        self.ConvBlock2 = block(channels, channels * 2, strides=1)
        self.pool2 = nn.Conv2d(channels * 2, channels * 2, kernel_size=5, stride=2, padding=2)
        
        self.ConvBlock3 = block(channels * 2, channels * 4, strides=1)
        self.pool3 = nn.Conv2d(channels * 4, channels * 4, kernel_size=5, stride=2, padding=2)
        
        # self.ConvBlock4 = block(channels * 4, channels * 8, strides=1)
        # self.pool4 = nn.Conv2d(channels * 8, channels * 8, kernel_size=5, stride=2, padding=2)
        
        # self.ConvBlock5 = block(channels * 8, channels * 8, strides=1)
        

    def forward(self, x, reverse=False):
        xs = []
        xs.append(x)

        conv1 = self.ConvBlock1(x)
        pool1 = self.pool1(conv1)
        xs.append(pool1)
        
        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)
        xs.append(pool2)
        
        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)
        xs.append(pool3)
        
        # conv4 = self.ConvBlock4(pool3)
        # pool4 = self.pool4(conv4)
        # xs.append(pool4)   
    
        return xs, reverse
