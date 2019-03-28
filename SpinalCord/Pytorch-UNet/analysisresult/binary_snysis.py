import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
from data.datautils import RandomCrop, CenterCrop
import math


import os
from scipy.misc import imsave 
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.pgm','.PGM'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def binary_relation_map(image):
    images_from = np.array(image,dtype=np.float32)
    images_pad = np.pad(images_from,((1,1),(1,1)),'constant')
    images_to = np.zeros((8,images_from.shape[0],images_from.shape[1])) #,constant_values = (0.0,0.0)
    images_to[0] = images_pad[:-2, 2:]
    images_to[1] = images_pad[1:-1,2:]
    images_to[2] = images_pad[2:, 2:]
    images_to[3] = images_pad[:-2,1:-1]
    images_to[4] = images_pad[2:,1:-1]
    images_to[5] = images_pad[:-2,:-2]
    images_to[6] = images_pad[1:-1,:-2]
    images_to[7] = images_pad[2:, :-2]
    diff_maps = images_to - images_from
    diff_maps[diff_maps>=0] = 1.0
    diff_maps[diff_maps<0] = 0.0
    relation_map = diff_maps
    return relation_map


def compute_relation_map(image):
    images_from = np.array(image,dtype=np.float32)
    images_pad = np.pad(images_from,((1,1),(1,1)),'constant')
    images_to = np.zeros((8,images_from.shape[0],images_from.shape[1])) #,constant_values = (0.0,0.0)
    images_to[0] = images_pad[:-2, 2:]
    images_to[1] = images_pad[1:-1,2:]
    images_to[2] = images_pad[2:, 2:]
    images_to[3] = images_pad[:-2,1:-1]
    images_to[4] = images_pad[2:,1:-1]
    images_to[5] = images_pad[:-2,:-2]
    images_to[6] = images_pad[1:-1,:-2]
    images_to[7] = images_pad[2:, :-2]
    diff_maps = images_to - images_from
    relation_map = (diff_maps - diff_maps.min()) / (diff_maps.max()-diff_maps.min()+1e-10) * 2.0
    relation_map = relation_map - 1.0
    # print(relation_map.max(), relation_map.min())
    # relation_map = diff_maps.copy()
    return relation_map


def imgtransform(image, name):
    # normalize image 
    # image = anisodiff(image, 4, 0.1, 0.1)
    image = (image - image.mean())/(image.std()+1e-10)
    image = (image - image.min()) / (image.max()-image.min()+1e-10) * 2.0
    image = image - 1.0
    return image

# path = "/home/jjchu/MyResearch/Seg/spinalcord/CGAN/results/cgan_80_64_10L1_10PER_resnet_5blocks_batch4_teststep300_binary_pad_meanstd/test_latest/images/site4-sc09-mask-r4_2_real.pgm"
# path = "/home/jjchu/MyResearch/Seg/spinalcord/CGAN/results/cgan_80_64_10L1_10PER_resnet_5blocks_batch4_teststep300_binary_pad_meanstd/test_latest/images/site4-sc09-mask-r4_2_fake.pgm"
path = "/home/jjchu/MyResearch/Seg/spinalcord/CGAN/results/cgan_80_64_10L1_10PER_resnet_5blocks_batch4_teststep300_binary_pad_meanstd/test_latest/images/site3-sc08-mask-r2_10_real.pgm"
# path = "/home/jjchu/MyResearch/Seg/spinalcord/CGAN/results/cgan_80_64_10L1_10PER_resnet_5blocks_batch4_teststep300_binary_pad_meanstd/test_latest/images/site3-sc08-mask-r2_10_fake.pgm"
name = path.split('/')[-1]
spath = "/home/jjchu/MyResearch/Seg/spinalcord/Pytorch-UNet/test/"
img = Image.open(path).convert('L')
img.save(os.path.join(spath,name))
img = np.array(img)
img = imgtransform(img, name)
aff1 = binary_relation_map(img)
aff2 = compute_relation_map(img)
for i in range(8):
    aff = np.array(aff1[i]*255,dtype=np.uint8)
    scipy.misc.imsave(os.path.join(spath,name+'_binary_'+str(i)+'.pgm'),aff)


for i in range(8):
    aff = np.array(aff2[i]*255,dtype=np.uint8)
    scipy.misc.imsave(os.path.join(spath,name+'_relation_'+str(i)+'.pgm'),aff)


