import torch
import torch.nn as nn
import torch.nn.functional as F

from ..qrnn3d.layer import ConvBlock
from .DCN import DCN_sep_pre_multi_offset_flow_similarity as DynAgg
from .self_cross_pos import spectralAttentionPositionLayer

class HSIDecoder(nn.Module):
    def __init__(self, channels, out_channel=31, block=ConvBlock, has_ad=True, bn=False, act='tanh'):
        super(HSIDecoder, self).__init__()
        print('HSIDecode!!!!')
        # Decoder
        # dynamic aggregation module for relu3_1 reference feature
        self.up_small_offset_conv1 = nn.Conv2d(
            channels + 256*2, 256, 3, 1, 1, bias=True)  # concat for diff
        self.up_small_offset_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=True)
        self.up_small_dyn_agg = DynAgg(256, 256, 3, stride=1, padding=1, dilation=1,
                                deformable_groups=8, extra_offset_mask=True)

        # self-cross attention
        # for small scale restoration
        self.res1 = spectralAttentionPositionLayer(inp_channels=channels, dim=256, out_channels=128, num_blocks=4, heads=8, upsample=True)

        # dynamic aggregation module for relu2_1 reference feature
        self.medium_fusion = nn.Conv2d(128*2, 128, 1, 1)
        self.up_medium_offset_conv1 = nn.Conv2d(
            channels//2 + 128*2, 128, 3, 1, 1, bias=True)
        self.up_medium_offset_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.up_medium_dyn_agg = DynAgg(128, 128, 3, stride=1, padding=1, dilation=1,
                                deformable_groups=8, extra_offset_mask=True)

        # for medium scale restoration
        self.res2 = spectralAttentionPositionLayer(inp_channels=128+128, dim=128, out_channels=64, num_blocks=3, heads=4, upsample=True)

        # dynamic aggregation module for relu1_1 reference feature
        self.large_fusion = nn.Conv2d(64*2, 64, 1, 1)
        self.up_large_offset_conv1 = nn.Conv2d(channels//4 + 64*2, 64, 3, 1, 1, bias=True)
        self.up_large_offset_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.up_large_dyn_agg = DynAgg(64, 64, 3, stride=1, padding=1, dilation=1,
                                deformable_groups=8, extra_offset_mask=True)

        # for large scale
        self.res3 = spectralAttentionPositionLayer(inp_channels=64+64, dim=64, out_channels=out_channel, num_blocks=2, heads=1, upsample=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
    def flow_warp(self,
                  x,
                  flow,
                  interp_mode='bilinear',
                  padding_mode='zeros',
                  align_corners=True):
        """Warp an image or feature map with optical flow.
        Args:
            x (Tensor): Tensor with size (n, c, h, w).
            flow (Tensor): Tensor with size (n, h, w, 2), normal value.
            interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
            padding_mode (str): 'zeros' or 'border' or 'reflection'.
                Default: 'zeros'.
            align_corners (bool): Before pytorch 1.3, the default value is
                align_corners=True. After pytorch 1.3, the default value is
                align_corners=False. Here, we use the True as default.
        Returns:
            Tensor: Warped image or feature map.
        """

        assert x.size()[-2:] == flow.size()[1:3]
        _, _, h, w = x.size()
        # create mesh grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h).type_as(x),
            torch.arange(0, w).type_as(x))
        grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
        grid.requires_grad = False

        vgrid = grid + flow
        # scale grid to [-1,1]
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(x,
                               vgrid_scaled,
                               mode=interp_mode,
                               padding_mode=padding_mode,
                               align_corners=align_corners)

        return output

    def forward(self, hsi_feats, ref_feats, offset):

        pre_offset = offset[0]
        pre_flow = offset[1]
        pre_similarity = offset[2]

        pre_relu1_swapped_feat = self.flow_warp(ref_feats['relu1_1'], pre_flow['relu1_1'])
        pre_relu2_swapped_feat = self.flow_warp(ref_feats['relu2_1'], pre_flow['relu2_1'])
        pre_relu3_swapped_feat = self.flow_warp(ref_feats['relu3_1'], pre_flow['relu3_1'])

        # dynamic aggregation for relu3_1 reference feature
        relu3_offset = torch.cat([hsi_feats[-1], pre_relu3_swapped_feat, ref_feats['relu3_1']], 1)
        relu3_offset = self.lrelu(self.up_small_offset_conv1(relu3_offset))
        relu3_offset = self.lrelu(self.up_small_offset_conv2(relu3_offset))
        relu3_swapped_feat = self.lrelu(
            self.up_small_dyn_agg([ref_feats['relu3_1'], relu3_offset], pre_offset['relu3_1'], pre_similarity['relu3_1']))

        # small scale
        x = self.res1(hsi_feats[-1], relu3_swapped_feat)

        # dynamic aggregation for relu2_1 reference feature
        relu2_fusion = self.medium_fusion(torch.cat([hsi_feats[-2], x], 1))
        relu2_offset = torch.cat([relu2_fusion, pre_relu2_swapped_feat, ref_feats['relu2_1']], 1)
        relu2_offset = self.lrelu(self.up_medium_offset_conv1(relu2_offset))
        relu2_offset = self.lrelu(self.up_medium_offset_conv2(relu2_offset))
        relu2_swapped_feat = self.lrelu(
            self.up_medium_dyn_agg([ref_feats['relu2_1'], relu2_offset],
                                pre_offset['relu2_1'], pre_similarity['relu2_1']))
        # medium scale
        x = self.res2(hsi_feats[-2], relu2_swapped_feat, x)

        # dynamic aggregation for relu1_1 reference feature
        relu1_fusion = self.large_fusion(torch.cat([hsi_feats[-3], x], 1))
        relu1_offset = torch.cat([relu1_fusion, pre_relu1_swapped_feat, ref_feats['relu1_1']], 1)
        relu1_offset = self.lrelu(self.up_large_offset_conv1(relu1_offset))
        relu1_offset = self.lrelu(self.up_large_offset_conv2(relu1_offset))
        relu1_swapped_feat = self.lrelu(
            self.up_large_dyn_agg([ref_feats['relu1_1'], relu1_offset],
                               pre_offset['relu1_1'], pre_similarity['relu1_1']))
        # large scale
        x = self.res3(hsi_feats[-3], relu1_swapped_feat, x)

        out = x + hsi_feats[-4]
        return out
