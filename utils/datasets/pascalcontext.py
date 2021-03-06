###########################################################################
# Created by: NTU EEE
# Email: ding0093@e.ntu.edu.sg
# Copyright (c) 2019
###########################################################################

import os
import sys
import numpy as np
import random
import math
from PIL import Image, ImageOps, ImageFilter, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

import torch
import torch.utils.data as data
import torchvision.transforms as transform
import re
from tqdm import tqdm
from .base import BaseDataset

class PascalContextSegmentation(BaseDataset):
    BASE_DIR = 'pascalcontext'
    NUM_CLASS = 59
    def __init__(self, root='datasets', split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(PascalContextSegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        print("split hhding: %s" % split)
        # assert exists
        root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(root), "Please download the dataset!!"

        self.images, self.masks = _get_pascalcontext_pairs(root, split)
        if split != 'vis':
            assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'vis':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        
        mask = Image.open(self.masks[index])
        
        # synchrosized transform
        if self.mode == 'train':
            img, mask, maskb = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        if self.mode == 'train':
            return img, mask, maskb
        else:
            return img, mask

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def _mask2maskb(self, mask):
        maskb = np.array(mask).astype('int32')
        maskb [maskb == 255] = -1
        maskb_ = np.array(mask).astype('float32')
        kernel = np.ones((9,9),np.float32)/81
        mask_tmp = cv2.filter2D(maskb_,-1, kernel)
        mask_tmp = abs(mask_tmp - maskb_)
        mask_tmp = mask_tmp > 0.005
        maskb[mask_tmp] = 59

        return torch.from_numpy(maskb).long()

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0


def _get_pascalcontext_pairs(folder, split='train'):
    def get_path_pairs(folder,split_f):
        img_paths = []
        mask_paths = []
        with open(split_f, 'r') as lines:
            for line in tqdm(lines):
                ll_str = re.split('\t', line)
                imgpath = os.path.join('/home/hhding/datasets/VOC2010/',ll_str[0].rstrip())
                maskpath = os.path.join('/home/hhding/datasets/VOC2010/',ll_str[1].rstrip())
                if os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask:', maskpath)
        return img_paths, mask_paths
    if split == 'train':
        split_f = os.path.join(folder, 'train.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    elif split == 'val':
        split_f = os.path.join(folder, 'val.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    return img_paths, mask_paths
