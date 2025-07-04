import os
from pathlib import Path

import imageio
import numpy as np
import torch.utils.data as data
from hdf5storage import loadmat
from matplotlib.image import imread
from torchlight.transforms.functional import minmax_normalize
from torchlight.transforms.stateful import RandCrop
from torchlight.transforms.general import CenterCrop

from .transform import SRDegrade


def imread(path):
    img = imageio.imread(path)
    img = np.array(img).astype('float')
    # img = img / 255
    return img


def hwc2chw(img):
    return img.transpose(2, 0, 1)


class FusionDataset(data.Dataset):
    """ Generate LR on demand 
        Require:
            - HSI HR
            - HSI RGB HR
            - Ref RGB HR
    """

    def __init__(self, root, input, ref, names_path, sf, 
                 mat_key='gt', crop_size=None, repeat=1, use_cache=False, sr_input=None, type=None):
        root = Path(root)
        self.hsi_inp_dir = root / (input+'_hsi') / 'HR'
        self.rgb_dir = root / (ref+'_cmatch') / 'HR'
        self.mat_key = mat_key
        names_path = root / names_path

        # get the name of all images
        if names_path is None:
            self.names = os.listdir(os.path.join(self.hsi_inp_dir))
        else:
            with open(names_path, 'r') as f:
                names = f.readlines()
                self.names = [n.strip() for n in names]

        self.loadmat = loadmat
        self.imread = imread
        print("scale factor: {}".format(sf))
        self.degrade = SRDegrade(sf)
        self.sf = sf
        
        self.crop_size = crop_size
        self.repeat = repeat
        self.names = self.names * repeat
        
        self.use_cache = use_cache
        self.cache = {}
        
        # if not none, use results in sr_input as the hsi_lr
        self.sr_input = sr_input
        if self.sr_input is None:
            print('There is no sr input!')
            
        self.type = type
        
    def __len__(self):
        return len(self.names) 

    def get_mat(self, path):
        # if self.use_cache:
        #     if path not in self.cache:
        #         self.cache[path] = self.loadmat(path)
        #     return self.cache[path]
        return self.loadmat(path)

    def __getitem__(self, index):
        name = self.names[index]

        hsi_hr = self.get_mat(str(self.hsi_inp_dir / (name+'.mat')))[self.mat_key]
        hsi_hr_flow = hsi_hr
        hsi_hr = minmax_normalize(hsi_hr.astype('float'))
        if self.sr_input:
            hsi_lr = self.get_mat(os.path.join(self.sr_input, name+'.mat'))['sspsr'].transpose(1,2,0)
            hsi_lr_flow = hsi_lr
        else:
            hsi_lr = self.degrade(hsi_hr)
            hsi_lr_flow = self.degrade(hsi_hr_flow)
            # preprocess hsi_lr and hsi_lr_flow for fewer training time
            # hsi_lr = self.get_mat(os.path.join('/media/exthdd2/code/SuperResolution/SSC-HSR/data/real-lr/sf'+str(self.sf), 'lr', name+'.mat'))['lr']
            # hsi_lr_flow = self.get_mat(os.path.join('/media/exthdd2/code/SuperResolution/SSC-HSR/data/real-lr/sf'+str(self.sf), 'flow', name+'.mat'))['lr']
            # hsi_lr_flow = minmax_normalize(hsi_lr_flow.astype('float')) * 255

        # hsi_rgb_hr = hsi_hr @ srf.T
        # hsi_rgb_lr = hsi_lr @ srf.T
        hsi_rgb_hr = hsi_hr[:,:,(15,8,3)].astype(np.float32)
        hsi_rgb_lr = hsi_lr[:,:,(15,8,3)].astype(np.float32)
        hsi_rgb_lr_flow = hsi_lr_flow[:,:,(15,8,3)].astype(np.float32)

        rgb_hr = self.imread(str(self.rgb_dir / (name+'.png')))
        rgb_hr_flow = rgb_hr
        rgb_hr = rgb_hr_flow / 255

        rgb_lr = self.degrade(rgb_hr)
        
        from skimage import exposure
        hsi_rgb_hr = exposure.match_histograms(hsi_rgb_hr, rgb_hr, multichannel=True)
        hsi_rgb_lr = exposure.match_histograms(hsi_rgb_lr, rgb_lr, multichannel=True)

        output = hsi_hr, hsi_lr, hsi_rgb_hr, hsi_rgb_lr, rgb_hr, rgb_lr, hsi_rgb_lr_flow, rgb_hr_flow
        output = tuple(hwc2chw(o) for o in output)

        if self.crop_size:
            if self.type == 'test':
                crop_fn = CenterCrop(self.crop_size)
                output = tuple(crop_fn(o) for o in output)
            else:
                H = output[0].shape[1]
                W = output[0].shape[2]
                crop_fn = RandCrop((H,W),self.crop_size)
                output = tuple(crop_fn(o) for o in output)
            
        hsi_hr, hsi_lr, hsi_rgb_hr, hsi_rgb_lr, rgb_hr, rgb_lr, hsi_rgb_lr_flow, rgb_hr_flow = output

        hsi_lr = np.clip(hsi_lr, 0, 1)
        hsi_rgb_lr = np.clip(hsi_rgb_lr, 0, 1)
        rgb_lr = np.clip(rgb_lr, 0, 1)

        # 3D卷积需要5维，2D卷积则不需要
        # hsi_hr = hsi_hr[None]
        # hsi_lr = hsi_lr[None]
        # rgb_hsi_hr = rgb_hsi_hr[None]
        return hsi_hr, hsi_lr, hsi_rgb_hr, hsi_rgb_lr, rgb_hr, rgb_lr, hsi_rgb_lr_flow, rgb_hr_flow