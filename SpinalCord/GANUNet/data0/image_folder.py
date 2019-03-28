###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path
import numbers
import numpy as np 


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.pgm','.PGM'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname): # use site1 and site3 for training generator
            	# if fname.startswith('site') or fname.startswith('site') :
               	if fname.startswith('site3') or fname.startswith('site4') :
                # if fname.startswith('site1') or fname.startswith('site2') :
                #     if int(fname[8:10])>=6:
                    path = os.path.join(root, fname)
                    images.append(path)
    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


def CenterCrop(imgarr, size):
    h, w, c = imgarr.shape
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        size = size
    th, tw = size

    img_left = int(round((w - tw) / 2.))
    img_top = int(round((h - th) / 2.))

    container = np.zeros((th, tw, imgarr.shape[-1]), np.float32)
    container= imgarr[img_top:img_top+th, img_left:img_left+tw]

    return container


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
