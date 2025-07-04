import torch
import torch.nn as nn
import torch.nn.functional as FF
import numpy as np

from functools import partial

from .conv import *

# 去掉QRNN和BiQRNN中的rnn

# """F pooling"""
# class QRNN3DLayer(nn.Module):
#     def __init__(self, in_channels, hidden_channels, conv_layer, act='tanh'):
#         super(QRNN3DLayer, self).__init__()
#         self.in_channels = in_channels
#         self.hidden_channels = hidden_channels
#         # quasi_conv_layer
#         self.conv = conv_layer
#         self.act = act

#     def _conv_step(self, inputs):
#         Z = self.conv(inputs)
#         if self.act == 'tanh':
#             return Z.tanh()
#         elif self.act == 'relu': 
#             return Z.relu()
#         elif self.act == 'none':
#             return Z
#         else:
#             raise NotImplementedError

#     def _rnn_step(self, z, f, h):
#         # uses 'f pooling' at each time step
#         h_ = (1 - f) * z if h is None else f * h + (1 - f) * z
#         return h_

#     def forward(self, inputs, reverse=False):
#         Z = self._conv_step(inputs)
#         return Z

#     def extra_repr(self):
#         return 'act={}'.format(self.act)


# class BiQRNN3DLayer(QRNN3DLayer):
#     def _conv_step(self, inputs):
#         Z = self.conv(inputs)
#         if self.act == 'tanh':
#             return Z.tanh()
#         elif self.act == 'relu':
#             return Z.relu()
#         elif self.act == 'none':
#             return Z
#         else:
#             raise NotImplementedError

#     def forward(self, inputs, fname=None):    
#         Z = self._conv_step(inputs)
#         return Z


# class BiQRNNConv3D(BiQRNN3DLayer):
#     def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bn=True, act='tanh'):
#         super(BiQRNNConv3D, self).__init__(
#             in_channels, hidden_channels, BasicConv3d(in_channels, hidden_channels, k, s, p, bn=bn), act=act)


# class BiQRNNDeConv3D(BiQRNN3DLayer):
#     def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bias=False, bn=True, act='tanh'):
#         super(BiQRNNDeConv3D, self).__init__(
#             in_channels, hidden_channels, BasicDeConv3d(in_channels, hidden_channels, k, s, p, bias=bias, bn=bn), act=act)

# class BiQRNNUpsampleConv3d(BiQRNN3DLayer):
#     def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, upsample=(1,2,2), bn=True, act='tanh'):
#         super(BiQRNNUpsampleConv3d, self).__init__(
#             in_channels, hidden_channels, BasicUpsampleConv3d(in_channels, hidden_channels, k, s, p, upsample, bn=bn), act=act)

# class QRNNConv3D(QRNN3DLayer):
#     def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bn=True, act='tanh'):
#         super(QRNNConv3D, self).__init__(
#             in_channels, hidden_channels, BasicConv3d(in_channels, hidden_channels, k, s, p, bn=bn), act=act)


# class QRNNDeConv3D(QRNN3DLayer):
#     def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bn=True, act='tanh'):
#         super(QRNNDeConv3D, self).__init__(
#             in_channels, hidden_channels, BasicDeConv3d(in_channels, hidden_channels, k, s, p, bn=bn), act=act)


# class QRNNUpsampleConv3d(QRNN3DLayer):
#     def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, upsample=(1,2,2), bn=True, act='tanh'):
#         super(QRNNUpsampleConv3d, self).__init__(
#             in_channels, hidden_channels, BasicUpsampleConv3d(in_channels, hidden_channels, k, s, p, upsample, bn=bn), act=act)


# class ConvBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, strides=1):
#         super(ConvBlock, self).__init__()
#         self.strides = strides
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
#             nn.LeakyReLU(inplace=True),
#         )
#         self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

#     def forward(self, x):
#         out1 = self.block(x)
#         out2 = self.conv11(x)
#         out = out1 + out2
#         return out

#     def flops(self, H, W):
#         flops = H * W * self.in_channel * self.out_channel * (3 * 3 + 1) + H * W * self.out_channel * self.out_channel * 3 * 3
#         return flops


# 去掉rnn，但计算的时候保留其他
"""F pooling"""
class QRNN3DLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, conv_layer, act='tanh'):
        super(QRNN3DLayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        # quasi_conv_layer
        self.conv = conv_layer
        self.act = act

    def _conv_step(self, inputs):
        gates = self.conv(inputs)
        Z, F = gates.split(split_size=self.hidden_channels, dim=1)
        if self.act == 'tanh':
            return Z.tanh(), F.sigmoid()
        elif self.act == 'relu': 
            return Z.relu(), F.sigmoid()
        elif self.act == 'none':
            return Z, F.sigmoid
        else:
            raise NotImplementedError

    def _rnn_step(self, z, f, h):
        # uses 'f pooling' at each time step
        h_ = (1 - f) * z if h is None else f * h + (1 - f) * z
        return h_

    def forward(self, inputs, reverse=False):
        h = None
        Z, F = self._conv_step(inputs)
        # h_time = []
        
        # if not reverse:
        #     for time, (z, f) in enumerate(zip(Z.split(1, 2), F.split(1, 2))):  # split along timestep            
        #         h = self._rnn_step(z, f, h)
        #         h_time.append(h)
        # else:
        #     for time, (z, f) in enumerate((zip(
        #         reversed(Z.split(1, 2)), reversed(F.split(1, 2))
        #         ))):  # split along timestep
        #         h = self._rnn_step(z, f, h)
        #         h_time.insert(0, h)
        
        # # return concatenated hidden states
        # return torch.cat(h_time, dim=2)
        return Z

    def extra_repr(self):
        return 'act={}'.format(self.act)


class BiQRNN3DLayer(QRNN3DLayer):
    def _conv_step(self, inputs):
        gates = self.conv(inputs)
        Z, F1, F2 = gates.split(split_size=self.hidden_channels, dim=1)
        if self.act == 'tanh':
            return Z.tanh(), F1.sigmoid(), F2.sigmoid()
        elif self.act == 'relu':
            return Z.relu(), F1.sigmoid(), F2.sigmoid()
        elif self.act == 'none':
            return Z, F1.sigmoid(), F2.sigmoid()
        else:
            raise NotImplementedError

    def forward(self, inputs, fname=None):        
        h = None 
        Z, F1, F2 = self._conv_step(inputs)
        # hsl = [] ; hsr = []
        # zs = Z.split(1, 2)

        # for time, (z, f) in enumerate(zip(zs, F1.split(1, 2))):  # split along timestep            
        #     h = self._rnn_step(z, f, h)
        #     hsl.append(h)
        
        # h = None
        # for time, (z, f) in enumerate((zip(
        #     reversed(zs), reversed(F2.split(1, 2))
        #     ))):  # split along timestep
        #     h = self._rnn_step(z, f, h)
        #     hsr.insert(0, h)
        
        # # return concatenated hidden states
        # hsl = torch.cat(hsl, dim=2)
        # hsr = torch.cat(hsr, dim=2)

        # if fname is not None:
        #     stats_dict = {'z':Z, 'fl':F1, 'fr':F2, 'hsl':hsl, 'hsr':hsr}
        #     torch.save(stats_dict, fname)
        # return hsl + hsr
        return Z


class BiQRNNConv3D(BiQRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bn=True, act='tanh'):
        super(BiQRNNConv3D, self).__init__(
            in_channels, hidden_channels, BasicConv3d(in_channels, hidden_channels*3, k, s, p, bn=bn), act=act)


class BiQRNNDeConv3D(BiQRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bias=False, bn=True, act='tanh'):
        super(BiQRNNDeConv3D, self).__init__(
            in_channels, hidden_channels, BasicDeConv3d(in_channels, hidden_channels*3, k, s, p, bias=bias, bn=bn), act=act)

class BiQRNNUpsampleConv3d(BiQRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, upsample=(1,2,2), bn=True, act='tanh'):
        super(BiQRNNUpsampleConv3d, self).__init__(
            in_channels, hidden_channels, BasicUpsampleConv3d(in_channels, hidden_channels*3, k, s, p, upsample, bn=bn), act=act)

class QRNNConv3D(QRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bn=True, act='tanh'):
        super(QRNNConv3D, self).__init__(
            in_channels, hidden_channels, BasicConv3d(in_channels, hidden_channels*2, k, s, p, bn=bn), act=act)


class QRNNDeConv3D(QRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bn=True, act='tanh'):
        super(QRNNDeConv3D, self).__init__(
            in_channels, hidden_channels, BasicDeConv3d(in_channels, hidden_channels*2, k, s, p, bn=bn), act=act)


class QRNNUpsampleConv3d(QRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, upsample=(1,2,2), bn=True, act='tanh'):
        super(QRNNUpsampleConv3d, self).__init__(
            in_channels, hidden_channels, BasicUpsampleConv3d(in_channels, hidden_channels*2, k, s, p, upsample, bn=bn), act=act)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out

    def flops(self, H, W):
        flops = H * W * self.in_channel * self.out_channel * (3 * 3 + 1) + H * W * self.out_channel * self.out_channel * 3 * 3
        return flops
