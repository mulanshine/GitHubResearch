import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import math 
import random

"""lblcgan_80_64_dataset.py"""

def f(lam,b):
    return np.exp(-1* (np.power(lam,2))/(np.power(b,2)))

def anisodiff(im, steps, b, lam = 0.25):  #takes image input, the number of iterations, 
    im_new = np.zeros(im.shape, dtype=im.dtype) 
    for t in range(steps): 
        dn = im[:-2,1:-1] - im[1:-1,1:-1] 
        ds = im[2:,1:-1] - im[1:-1,1:-1] 
        de = im[1:-1,2:] - im[1:-1,1:-1] 
        dw = im[1:-1,:-2] - im[1:-1,1:-1] 
        im_new[1:-1,1:-1] = im[1:-1,1:-1] +\
                            lam * (f(dn,b)*dn + f (ds,b)*ds + 
                                    f (de,b)*de + f (dw,b)*dw) 
        im = im_new 
    return im


# use(80,48),resize to (80,48) for generator
class LblResCGANDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.phase = opt.phase
        self.imgdir = os.path.join(opt.dataroot,'crop_128/image_crop/')
        self.maskdir = os.path.join(opt.dataroot,'crop_128/mask_crop/')
        self.imgpaths = make_dataset(self.imgdir,opt.site)
        self.imgpaths = sorted(self.imgpaths)
        self.valid_classes = [0, 128, 255]
        self.train_classes = [0, 1, 2]
        self.class_map = dict(zip(self.valid_classes, self.train_classes))
        if self.phase == 'train':
            self.targetimgpaths = sorted(make_dataset(self.imgdir,opt.target_site))
            self.t_size = len(self.targetimgpaths)

    def __getitem__(self, index):
        imgpath = self.imgpaths[index]
        name = imgpath.split('/')[-1]
        maskpath = os.path.join(self.maskdir,name)
        mask = Image.open(maskpath).convert('L')
        # imgL = Image.open(imgpath).convert('L')
        imgRGB = Image.open(imgpath).convert('RGB')
        # imgL = self.imgtransform_L(imgL,name)
        mask = self.imgtransform_mask(mask,name)
        imgRGB = self.imgtransform_RGB(imgRGB,name)
        if self.phase == 'train':
            index_t = random.randint(0, self.t_size - 1)
            targetimgpath = self.targetimgpaths[index_t]
            targetname = targetimgpath.split('/')[-1]
            target = Image.open(targetimgpath).convert('RGB')
            target = self.imgtransform_RGB(target,targetname)
        else:
            target = imgRGB
        # aff = self.binary_relation_map(imgL)
        # aff = self.compute_relation_map(imgL)
        return {'real': imgRGB,'target':target, 'label':mask, 'path': imgpath}

    def encode_segmap(self, mask):
        for validc in self.valid_classes:
            mask[mask==validc] = self.class_map[validc]
        return mask


    def imgtransform_L(self, image, name):
        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        # normalize image 
        image = (image - 0.5)*2
        # normalize image      
        image_shape = image.shape
        # print(image_shape)
        # image = np.expand_dims(image, axis=0)
        image = torch.FloatTensor(image)
        return image


    def imgtransform_mask(self, label, name):
        label = self.encode_segmap(np.array(label, dtype=np.uint8))
        label = np.expand_dims(label, axis=0)
        label = torch.FloatTensor(label)
        # label = label.long()
        return label

    def imgtransform_RGB(self, image, name):
        image = np.array(image, dtype=np.float32)
        # normalize image
        image = image / 255.0
        image = image *2 - 1.0
        image_shape = image.shape
        # print(image_shape)
        image = image.transpose((2,0,1))
        # image = np.expand_dims(image, axis=0)
        image = torch.FloatTensor(image)
        return image

    def binary_relation_map(self,image):
        images_from = np.array(image,dtype=np.float32)[0]
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
        relation_map = torch.FloatTensor(diff_maps)
        return relation_map

    def compute_relation_map(self, image):
        images_from = np.array(image,dtype=np.float32)[0]
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
        relation_map = torch.FloatTensor(relation_map)
        # relation_map = diff_maps.copy()
        return relation_map

    def __len__(self):
        return len(self.imgpaths)

    def name(self):
        return 'LblResCGANDataset'
