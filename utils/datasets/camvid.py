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

class CamVidSegmentation(BaseDataset):
    BASE_DIR = 'CamVid'
    NUM_CLASS = 11
    def __init__(self, root='datasets', split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(CamVidSegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        print("split hhding: %s" % split)
        # assert exists
        root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(root), "Please download the dataset!!"

        self.images, self.masks = _get_CamVid_pairs(root, split)
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
        im_w=img.size[0]
        im_h=img.size[1]
        img = img.resize((int(im_w/2),int(im_h/2)), Image.BILINEAR)
        mask = mask.resize((int(im_w/2),int(im_h/2)), Image.NEAREST)
        
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

    def _sync_transform(self, img, mask):
        # im_w=img.size[0]
        # im_h=img.size[1]
        # img = img.resize((int(im_w/2),int(im_h/2)), Image.BILINEAR)
        # mask = mask.resize((int(im_w/2),int(im_h/2)), Image.NEAREST)
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        if self.scale:
            long_size = random.randint(int(self.base_size*0.75), int(self.base_size*2.0))## random size 0.75~2.0
        else:
            long_size = self.base_size
        w, h = img.size
        if h < w:
            ow = long_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = long_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # random rotate -10~10, mask using NN rotate
        # deg = random.uniform(-10, 10)
        # img = img.rotate(deg, resample=Image.BILINEAR)
        # mask = mask.rotate(deg, resample=Image.NEAREST)
        # pad crop
        short_size = min(ow, oh)
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)#pad 255 for cityscapes
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        # gaussian blur as in PSP
        # if random.random() < 0.5:
        #     img = img.filter(ImageFilter.GaussianBlur(
        #         radius=random.random()))
        # final transform
        return img, self._mask_transform(mask), self._mask2maskb(mask)

    def _val_sync_transform(self, img, mask):
        # im_w=img.size[0]
        # im_h=img.size[1]
        # img = img.resize((int(im_w/2),int(im_h/2)), Image.BILINEAR)
        # mask = mask.resize((int(im_w/2),int(im_h/2)), Image.NEAREST)
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1+outsize, y1+outsize))
        mask = mask.crop((x1, y1, x1+outsize, y1+outsize))

        ## final transform
        return img, self._mask_transform(mask)

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
        maskb[mask_tmp] = 11

        return torch.from_numpy(maskb).long()

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0


def _get_CamVid_pairs(folder, split='train'):
    def get_path_pairs(folder,split_f):
        img_paths = []
        mask_paths = []
        with open(split_f, 'r') as lines:
            for line in tqdm(lines):
                ll_str = re.split('\t', line)
                imgpath = os.path.join('/home/hhding/datasets/CamVid/',ll_str[0].rstrip())
                maskpath = os.path.join('/home/hhding/datasets/CamVid/',ll_str[1].rstrip())
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
