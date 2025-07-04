import flow_vis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchlight
from torchlight.metrics import mpsnr, mssim, sam
from torchlight.utils.helper import get_obj, to_device
from torchlight.nn.ops.gradient import image_gradients
import numpy as np

from ..model import refsr as model_zoo
from .util import cal_metric, listmap, squeeze2img, squeeze2img_rgb, squeeze2img_chikusei
from .loss import gradient_warping_loss

from thop import profile
from torchlight.utils.stat import print_model_size
from torch.nn.parallel import DistributedDataParallel


class BaseModule(torchlight.Module):
    def __init__(self, model, optimizer):
        super().__init__()
        self.device = torch.device('cuda')
        self.model = get_obj(model, model_zoo).to(self.device)
        self.optimizer = get_obj(optimizer, optim, self.model.parameters())
        self.criterion = nn.SmoothL1Loss()
        self.clip_max_norm = 1e6

    def step(self, data, train, epoch, step):
        if train:
            return self._train(data)
        return self._eval(data)

    def state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state):
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        print('load_state_dic!!!!!!!!!!!!')

    def get_network_description(self, network):
        """Get the string and total parameters of the network"""
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        return str(network), sum(map(lambda x: x.numel(), network.parameters()))
    
    def print_network(self, logger):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.model.__class__.__name__,
                                             self.model.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.model.__class__.__name__)
        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)


class CommonModule(BaseModule):
    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)

    def _train(self, data):
        self.model.train()
        self.optimizer.zero_grad()

        data = to_device(data, self.device)
        hsi_hr, hsi_lr, hsi_rgb_hr, hsi_rgb_lr, rgb_hr, rgb_lr = data
        target = hsi_hr
        output, _, _, _ = self.model(hsi_sr=hsi_lr, hsi_rgb_sr=hsi_rgb_lr, ref_hr=rgb_hr)

        loss = self.criterion(output, target)

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_max_norm)
        self.optimizer.step()

        metrics = {'loss': loss.item(),
                   'psnr': cal_metric(mpsnr, output, target),
                   }

        return self.StepResult(metrics=metrics)

    def _eval(self, data):
        data = to_device(data, self.device)
        hsi_hr, hsi_lr, hsi_rgb_hr, hsi_rgb_lr, rgb_hr, rgb_lr = data
        target = hsi_hr
        output, warped_rgb, flow, masks = self.model(hsi_sr=hsi_lr, hsi_rgb_sr=hsi_rgb_lr, ref_hr=rgb_hr)

        metrics = {'psnr': cal_metric(mpsnr, output, target),
                   'ssim': cal_metric(mssim, output, target),
                   'sam': cal_metric(sam, output, target),
                   }
        
        return self.StepResult(metrics=metrics)


class SimpleModule(BaseModule):
    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)

    def _train(self, data):
        self.model.train()
        self.optimizer.zero_grad()

        data = to_device(data, self.device)
        hsi_hr, hsi_lr, hsi_rgb_hr, hsi_rgb_lr, rgb_hr, rgb_lr = data
        target = hsi_hr
        output = self.model(hsi_sr=hsi_lr, hsi_rgb_sr=hsi_rgb_lr, ref_hr=rgb_hr)

        loss = self.criterion(output, target)

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_max_norm)
        self.optimizer.step()

        metrics = {'loss': loss.item(), 'psnr': cal_metric(mpsnr, output, target)}
        return self.StepResult(metrics=metrics)

    def _eval(self, data):
        data = to_device(data, self.device)
        hsi_hr, hsi_lr, hsi_rgb_hr, hsi_rgb_lr, rgb_hr, rgb_lr = data
        target = hsi_hr
        output = self.model(hsi_sr=hsi_lr, hsi_rgb_sr=hsi_rgb_lr, ref_hr=hsi_rgb_hr)

        metrics = {'psnr': cal_metric(mpsnr, output, target),
                   'ssim': cal_metric(mssim, output, target),
                   'sam': cal_metric(sam, output, target)}
        
        return self.StepResult(metrics=metrics)


class FusionModule(BaseModule):
    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)

    def _train(self, data):
        self.model.train()
        self.optimizer.zero_grad()

        data = to_device(data, self.device)
        hsi_hr, hsi_lr, hsi_rgb_hr, hsi_rgb_lr, rgb_hr, rgb_lr, hsi_rgb_lr_flow, rgb_hr_flow = data
        target = hsi_hr
        output, warpnet_rgb, warped_rgb, flow = self.model(hsi_sr=hsi_lr, hsi_rgb_sr=hsi_rgb_lr, ref_hr=rgb_hr, hsi_rgb_sr_flow=hsi_rgb_lr_flow, ref_hr_flow=rgb_hr_flow)

        loss = self.criterion(output, target) + self.criterion(warpnet_rgb, hsi_rgb_hr)

        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_max_norm)
        self.optimizer.step()

        metrics = {'loss': loss.item(),
                   'psnr': cal_metric(mpsnr, output, target),
                   'norm': norm.item(),
                   'psnr_input': cal_metric(mpsnr, hsi_lr, target),
                   'flow': flow.max().item()
                   }

        return self.StepResult(metrics=metrics)

    def _eval(self, data):
        data = to_device(data, self.device)
        hsi_hr, hsi_lr, hsi_rgb_hr, hsi_rgb_lr, rgb_hr, rgb_lr, hsi_rgb_lr_flow, rgb_hr_flow = data
        target = hsi_hr
        output, warpnet_rgb, warped_rgb, flow = self.model(hsi_sr=hsi_lr, hsi_rgb_sr=hsi_rgb_lr, ref_hr=rgb_hr, hsi_rgb_sr_flow=hsi_rgb_lr_flow, ref_hr_flow=rgb_hr_flow)

        metrics = {'psnr': cal_metric(mpsnr, output, target),
                   'ssim': cal_metric(mssim, output, target),
                   'sam': cal_metric(sam, output, target),
                   'b-psnr': cal_metric(mpsnr, hsi_lr, target),
                   'b-ssim': cal_metric(mssim, hsi_lr, target),
                   'b-sam': cal_metric(sam, hsi_lr, target),
                   'flow': flow.max().item(),
                   }

        return self.StepResult(metrics=metrics)

    def visualize_mask(self, masks):
        interps = []
        sf = 1
        visuals = {}
        for idx, mask in enumerate(masks):
            visuals[f'mask{idx}'] = mask
            interp = F.interpolate(mask, scale_factor=sf)
            interp = ((interp-interp.min())/interp.max())[0]
            interps.append(interp)
            sf *= 2
        visuals['masks'] = interps
        return visuals
    