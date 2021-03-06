import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset,CenterCrop
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import math 


# center_crop 200

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
class SegcganDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.phase = opt.phase
        self.imgdir = os.path.join(opt.dataroot,'images')
        self.maskdir = os.path.join(opt.dataroot,'mask')
        self.maskpaths = make_dataset(self.maskdir)
        self.maskpaths = sorted(self.maskpaths)
        self.valid_classes = [0, 128, 255]
        self.train_classes = [0, 1, 2]
        self.crop_size = 200
        self.class_map = dict(zip(self.valid_classes, self.train_classes))

    def __getitem__(self, index):
        maskpath = self.maskpaths[index]
        mask_name = maskpath.split('/')[-1]
        img_name = "-".join(mask_name.replace("mask", "image").split('-')[:3]) + "_" + mask_name.split('_')[-1]
        imgpath = os.path.join(self.imgdir,img_name)
        mask = Image.open(maskpath).convert('L')
        imgL = Image.open(imgpath).convert('L')
        imgRGB = Image.open(imgpath).convert('RGB')
        imgL = self.imgtransform_L(imgL,img_name,self.crop_size)
        mask = self.imgtransform_mask(mask,mask_name,self.crop_size)
        imgRGB = self.imgtransform_RGB(imgRGB,img_name,self.crop_size)
        # aff = self.binary_relation_map(imgL)
        aff = self.compute_relation_map(imgL)
        return {'img': imgRGB, 'lbl':mask, 'aff':aff, 'path': imgpath}

    def encode_segmap(self, mask):
        for validc in self.valid_classes:
            mask[mask==validc] = self.class_map[validc]
        return mask


    def imgtransform_L(self, image, name,crop_size):
        size = np.array(image, dtype=np.uint8).shape
        if name.startswith('site1') or name.startswith('site2'):
            image = image.resize((math.ceil(2.0*size[1]),math.ceil(2.0*size[0])), Image.BICUBIC)
            image = np.asarray(image, np.float32)
        elif name.startswith('site4'):
            image = image.resize((math.ceil(1.16*size[1]),math.ceil(1.16*size[0])), Image.BICUBIC)
            image = np.asarray(image, np.float32)
        elif name.startswith('site3'):
            image = image.resize((math.ceil(1.0*size[1]),math.ceil(1.0*size[0])), Image.BICUBIC)
            image = np.array(image, dtype=np.float32)

        image = np.expand_dims(image, axis=2) 
        image = CenterCrop(image,crop_size)
        image = np.squeeze(image)
        image = image / 255.0
        # normalize image 
        # image = anisodiff(image, 4, 0.1, 0.1)
        image = (image - image.mean())/(image.std()+1e-10)
        image = (image - image.min()) / (image.max()-image.min()+1e-10) * 2.0
        image = image - 1.0
        # normalize image      
        image = np.expand_dims(image, axis=0)
        image = torch.FloatTensor(image)
        return image

    def imgtransform_mask(self, label, name, crop_size):
        size = np.array(label, dtype=np.uint8).shape
        if name.startswith('site1') or name.startswith('site2'):
            label = label.resize((math.ceil(2.0*size[1]),math.ceil(2.0*size[0])), Image.NEAREST)
            label = self.encode_segmap(np.array(label, dtype=np.uint8))
        elif name.startswith('site4'):
            label = label.resize((math.ceil(1.16*size[1]),math.ceil(1.16*size[0])), Image.NEAREST)
            label = self.encode_segmap(np.array(label, dtype=np.uint8))
        elif name.startswith('site3'):
            label = label.resize((math.ceil(1.0*size[1]),math.ceil(1.0*size[0])), Image.NEAREST)
            label = self.encode_segmap(np.array(label, dtype=np.uint8))
        
        label = np.expand_dims(label, axis=2) 
        label = CenterCrop(label,crop_size)
        label = np.squeeze(label)
        # label = np.expand_dims(label, axis=0)
        label = torch.FloatTensor(label)
        label = label.long()
        return label

    def imgtransform_RGB(self, image, name, crop_size):
        size = np.array(image).shape
        if name.startswith('site1') or name.startswith('site2'):
            image = image.resize((math.ceil(2*size[1]),math.ceil(2*size[0])), Image.BICUBIC)
            image = np.asarray(image, np.float32)
        elif name.startswith('site4'):
            image = image.resize((math.ceil(1.16*size[1]),math.ceil(1.16*size[0])), Image.BICUBIC)
            image = np.asarray(image, np.float32)
        elif name.startswith('site3'):
            image = image.resize((math.ceil(1.0*size[1]),math.ceil(1.0*size[0])), Image.BICUBIC)
            image = np.array(image, dtype=np.float32)

        image = CenterCrop(image,crop_size)
        image = np.squeeze(image)
        image = image / 255.0
        # normalize image 
        # image = anisodiff(image, 4, 0.1, 0.1)
        image = (image - image.mean())/(image.std()+1e-10)
        image = (image - image.min()) / (image.max()-image.min()+1e-10) * 2.0
        image = image - 1.0
        image = image.transpose((2,0,1))
        # normalize image      
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
        return len(self.maskpaths)

    def name(self):
        return 'SegcganDataset'

