from functools import partial

import torch.nn as nn
import torch

from .decoder import HSIDecoder
from .encoder import HSIEncoder
from .feature import ContrasExtractorSep
from .similar import FlowSimCorrespondenceGenerationArch
from torch.nn.parallel import DataParallel, DistributedDataParallel
from .warpnet import warpnet

class FusionWarpNetHSI(nn.Module):
    def __init__(self, reweight=False, use_mask=True, use_pwc=True, sample=False, sample_ratio=None, finetune=True):
        super().__init__()
        if not sample:
            self.flownet = self.flow_estimator(use_pwc)
        else:
            self.flownet = self.flow_estimator_pretrained(sample_ratio, finetune)
        self.warp = self.backwarp(use_pwc)
        self.warpnet = self.warpnet_load()

        self.hsi_encoder = HSIEncoder(31, 64)
        self.decoder = HSIDecoder(256)

        self.rgb_extractor = self.extractor()
        self.similar = FlowSimCorrespondenceGenerationArch(patch_size=3, stride=1, vgg_layer_list=['relu1_1', 'relu2_1', 'relu3_1'], vgg_type='vgg19')
    
    # RAFT
    def flow_estimator(self, use_pwc):
        from .raft import RAFT
        net = torch.nn.DataParallel(RAFT())
        net.load_state_dict(torch.load('/media/exthdd2/code/SuperResolution/SSC-HSR/source/model/refsr/SSCHSR/pretrained/raft-things.pth'))
        print("flow network RAFT load successfully!")
        net = net.module
        return net
    
    def flow_estimator_pretrained(self, sample_ratio, finetune):
        from .raft import RAFT
        net = torch.nn.DataParallel(RAFT())
        net.load_state_dict(torch.load('/media/exthdd2/code/SuperResolution/SSC-HSR/source/model/refsr/SSCHSR/pretrained/sf' + str(sample_ratio) + '/50000_raft-rgb-sf'+ str(sample_ratio) +'.pth'))
        print("flow network RAFT_pretrained_sf{} load successfully!".format(sample_ratio))
        net = net.module
        if not finetune:
            print("flow network RAFT_pretrained_sf{} will not finetune!".format(sample_ratio))
            for param in net.parameters():
                param.requires_grad = False
        return net

    def backwarp(self, use_pwc):
        from torchlight.nn.ops.warp import flow_warp
        return partial(flow_warp, padding_mode='border')
    
    def load_network(self, net, load_path, strict=True):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
        """
        if isinstance(net, nn.DataParallel) or isinstance(
                net, DistributedDataParallel):
            net = net.module
        load_net = torch.load(load_path)

        # remove unnecessary 'module.'
        for k, v in load_net.items():  # cjz
                if k.startswith('module.'):
                    load_net[k[7:]] = v
                    load_net.pop(k)
        net.load_state_dict(load_net, strict=strict)

    def extractor(self):
        net = ContrasExtractorSep()
        self.load_network(net, '/media/exthdd2/code/SuperResolution/SSC-HSR/source/model/refsr/SSCHSR/pretrained/feature_extraction.pth')

        return net
    
    def warpnet_load(self):
        net = warpnet()
        checkpoints = torch.load('/media/exthdd2/code/SuperResolution/SSC-HSR/source/model/refsr/SSCHSR/pretrained/sf8/warpnet-sf8.pth')
        checkpoint = checkpoints['module']['model']
        for k, v in list(checkpoint.items()):
            if k.startswith('flownet.'):
                checkpoint.pop(k)
            elif k.startswith('warpnet.'):
                checkpoint[k[8:]] = v
                checkpoint.pop(k)
        net.load_state_dict(checkpoint, strict=True)
        return net

    def forward(self, hsi_sr, hsi_rgb_sr, ref_hr, hsi_rgb_sr_flow, ref_hr_flow):
        # # optical flow estimation
        ref_warp, flow_12_1 = self.corase_align(hsi_rgb_sr, ref_hr, hsi_rgb_sr_flow, ref_hr_flow)
        ref_warpnet = self.warpnet(hsi_rgb_sr, ref_hr, ref_warp)
        features = self.rgb_extractor(hsi_rgb_sr, ref_warpnet)
        offset, ref_feats = self.similar(features, ref_warpnet)
        hsi_feats, reverse = self.hsi_encoder(hsi_sr)
        out = self.decoder(hsi_feats, ref_feats, offset)
        return out, ref_warpnet, ref_warp, flow_12_1

    def corase_align(self, x, ref, x_flow, ref_flow):
        # raft
        flow = self.flownet(x_flow, ref_flow)
        flow_12_1 = flow['flow_12_1']  # B, 2, W, H
        ref_warp = self.warp(ref, flow_12_1)
        return ref_warp, flow_12_1
