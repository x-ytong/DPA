from __future__ import print_function, division
import os
from torch.utils.data import Dataset
from torchvision import transforms

import torch
import random
import numpy as np
from PIL import Image, ImageFilter


class TRGData(Dataset):

    def __init__(self, args):
        super().__init__()
        self.imdr = os.path.join(args.target_dir, args.target)
        self.args = args

        self.im_ids = []
        self.images = []

        for id in os.listdir(self.imdr):
            im = os.path.join(self.imdr, id)
            assert os.path.isfile(im)
            self.im_ids.append(id)
            self.images.append(im)

        # Display stats
        print('Number of images in {}: {:d}'.format(args.target, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('CMYK')
        sample = {'image': _img}

        if self.args.target == 'wuhan':
            mean = (0.537, 0.375, 0.374, 0.370)
            std = (0.239, 0.237, 0.230, 0.229)
        elif self.args.target == 'chengdu':
            mean = (0.577, 0.362, 0.410, 0.372)
            std = (0.235, 0.245, 0.231, 0.230)
        elif self.args.target == 'shanghai':
            mean = (0.547, 0.415, 0.416, 0.391)
            std = (0.273, 0.265, 0.258, 0.256)
        elif self.args.target == 'beijing':
            mean = (0.520, 0.391, 0.416, 0.385)
            std = (0.247, 0.255, 0.241, 0.236)
        elif self.args.target == 'guangzhou':
            mean = (0.590, 0.286, 0.323, 0.308)
            std = (0.238, 0.238, 0.226, 0.225)
        else:
            raise

        composed_transforms = transforms.Compose([
            RandomHorizontalFlip(),
            RandomGaussianBlur(),
            Normalize(mean=mean, std=std),
            ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'Target (set = ' + self.args.target + ')'


class Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img}


class ToTensor(object):
    def __call__(self, sample):
        img = sample['image']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()

        return {'image': img}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img}



