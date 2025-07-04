import os
from functools import partial

import numpy as np
import torch
import torch.utils.data as data
from hdf5storage import loadmat
from torchlight.transforms import CenterCrop
from torchlight.transforms.stateful import RandCrop
from torchlight.transforms.functional import minmax_normalize

from .transform import SRDegrade

def hwc2chw(img):
    return img.transpose(2, 0, 1)

class Transform:
    def __init__(self, sf, use_2dconv):
        self.degrade = SRDegrade(sf)
        self.hsi2tensor = partial(hsi2tensor, use_2dconv=use_2dconv)

    def _get_lr_sr_(self, hr):
        tmp = hr.transpose(1, 2, 0)
        lr = self.degrade.down(tmp)
        sr = self.degrade.up(lr)
        lr = lr.transpose(2, 0, 1)
        sr = sr.transpose(2, 0, 1)
        return lr, sr

    def __call__(self, hr):
        hr = hr.astype('float')
        # hr = minmax_normalize(hr)
        lr, sr = self._get_lr_sr_(hr)
        lr, sr, hr = map(self.hsi2tensor, (lr, sr, hr))
        return lr, sr, hr

class RealDataset(data.Dataset):
    def __init__(self, root, input, names_path, sf, mat_key='gt', crop_size=None, repeat=1):
        from pathlib import Path
        root = Path(root)
        self.hsi_inp_dir = root / input
        self.mat_key = mat_key
        names_path = root / names_path
        if names_path is None:
            self.names = os.listdir(os.path.join(self.hsi_inp_dir))
        else:
            with open(names_path, 'r') as f:
                names = f.readlines()
                self.names = [n.strip() for n in names]
        self.loadmat = loadmat
        self.crop_size = crop_size
        self.repeat = repeat
        self.names = self.names * repeat

    def __getitem__(self, index):
        name = self.names[index]

        hsi_hr = self.get_mat(str(self.hsi_inp_dir / (name+'.mat')))[self.mat_key]
        # hsi_hr = minmax_normalize(hsi_hr.astype('float'))
        hsi_hr = hwc2chw(hsi_hr)
        if self.crop_size:
            H = hsi_hr.shape[1]
            W = hsi_hr.shape[2]
            crop_fn = RandCrop((H,W),self.crop_size)
            hsi_hr = crop_fn(hsi_hr)

        return hsi_hr

    def __len__(self):
        return len(self.names) 

    def get_mat(self,  path):
        return self.loadmat(path)

class SISRDataset(data.Dataset):
    def __init__(self, root, input, names_path, sf, mat_key='gt', crop_size=None, repeat=1, use_2dconv=True):
        super().__init__()
        self.dataset = RealDataset(root, input, names_path, sf, mat_key=mat_key, crop_size=crop_size, repeat=repeat)
        self.tsfm = Transform(sf, use_2dconv)

    def __getitem__(self, index):
        hr = self.dataset.__getitem__(index)
        lr, sr, hr = self.tsfm(hr)
        # print(lr.shape, sr.shape, hr.shape)
        return lr, sr, hr

    def __len__(self):
        return len(self.dataset)

# ---------------------------------------------------------------------------- #
#                                     Utils                                    #
# ---------------------------------------------------------------------------- #


def hsi2tensor(hsi, use_2dconv):
    """
    Transform a numpy array with shape (C, H, W)
    into torch 4D Tensor (1, C, H, W) or (C, H, W)
    """
    if use_2dconv:
        img = torch.from_numpy(hsi)
    else:
        img = torch.from_numpy(hsi[None])
    return img.float()


class MatDataFromFolder(data.Dataset):
    """Wrap mat data from folder"""

    def __init__(self, dataroot, load=loadmat, suffix='mat',
                 fns=None, size=None, attach_filename=False):
        super(MatDataFromFolder, self).__init__()
        self.load = load
        self.dataroot = dataroot

        if fns:
            with open(fns, 'r') as f:
                self.fns = [l.strip()+'.mat' for l in f.readlines()]
        else:
            self.fns = list(filter(lambda x: x.endswith(suffix), os.listdir(dataroot)))

        if size and size <= len(self.fns):
            self.fns = self.fns[:size]

        self.attach_filename = attach_filename

    def __getitem__(self, index):
        fn = self.fns[index]
        mat = self.load(os.path.join(self.dataroot, fn))
        if self.attach_filename:
            fn, _ = os.path.splitext(fn)
            fn = os.path.basename(fn)
            return mat, fn
        return mat

    def __len__(self):
        return len(self.fns)
