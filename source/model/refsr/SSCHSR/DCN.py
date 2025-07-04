#!/usr/bin/env python

import logging
import math

#import _ext as _backend
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
import pdb
import torch.utils.checkpoint as checkpoint
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d

class DCNv2(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 deformable_groups=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        assert 2 * self.deformable_groups * self.kernel_size[
            0] * self.kernel_size[1] == offset.shape[1]
        assert self.deformable_groups * self.kernel_size[0] * self.kernel_size[
            1] == mask.shape[1]
        return modulated_deform_conv2d(input, offset, mask, self.weight, self.bias,
                           self.stride, self.padding, self.dilation, 1,
                           self.deformable_groups)

class DCN_sep_pre_multi_offset_flow_similarity(DCNv2):
    '''
    Use other features to generate offsets and masks.

    Intialized the offset with precomputed non-local offset.
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 deformable_groups=1,
                 extra_offset_mask=True,
                 max_residue_magnitude=10,
                 use_sim=True
                 ):
        super(DCN_sep_pre_multi_offset_flow_similarity,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, deformable_groups)
        self.extra_offset_mask = extra_offset_mask
        self.max_residue_magnitude = max_residue_magnitude
        self.use_sim = use_sim

        channels_ = self.deformable_groups * 3 * self.kernel_size[
            0] * self.kernel_size[1]

        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            channels_,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True)

        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, x, pre_offset, pre_sim):
        '''
        Args:
            pre_offset: precomputed_offset. Size: [b, 9, h, w, 2]
        '''

        if self.extra_offset_mask:
            # x = [input, features] x[1]:[9, 256, 40, 40]
            out = self.conv_offset_mask(x[1])  # [9, 216, 40, 40]
            x = x[0]  # [9, 256, 40, 40]
        else:
            out = self.conv_offset_mask(x)

        o1, o2, mask = torch.chunk(out, 3, dim=1)   # [9, 72, 40, 40]
        offset = torch.cat((o1, o2), dim=1)  # [9, 144, 40, 40]
        if self.max_residue_magnitude:
            offset = self.max_residue_magnitude * torch.tanh(offset)
        # repeat pre_offset along dim1, shape: [b, 9*groups, h, w, 2]
        pre_offset = pre_offset.repeat([1, self.deformable_groups, 1, 1, 1])  #[9, 72, 40, 40, 2]
        # the order of offset is [y, x, y, x, ..., y, x]
        pre_offset_reorder = torch.zeros_like(offset)    # [9, 144, 40, 40]
        # add pre_offset on y-axis
        pre_offset_reorder[:, 0::2, :, :] = pre_offset[:, :, :, :, 1]
        # add pre_offset on x-axis
        pre_offset_reorder[:, 1::2, :, :] = pre_offset[:, :, :, :, 0]
        offset = offset + pre_offset_reorder  # [9, 144, 40, 40]

        if self.use_sim:
            mask = torch.sigmoid(mask*pre_sim)
        else:
            mask = torch.sigmoid(mask)  # [9, 72, 40, 40]

        offset_mean = torch.mean(torch.abs(offset - pre_offset_reorder))
        # import pdb
        # pdb.set_trace()
        if offset_mean > 100:
            print(
                'Offset mean is {}, larger than 100.'.format(offset_mean))

        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                           self.stride, self.padding, self.dilation, 1,
                           self.deformable_groups)
