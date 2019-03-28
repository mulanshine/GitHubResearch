import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import numbers
import math

# use orign image resize to the same space,and crop center 128*128 for generator
class ResizeCganDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.phase = opt.phase
        self.imgdir = os.path.join(opt.dataroot,"'images'")
        self.imgpaths = make_dataset(self.imgdir)
        self.imgpaths = sorted(self.imgpaths)

    def __getitem__(self, index):
        imgpath = self.imgpaths[index]
        image = Image.open(imgpath).convert('L')
        label = Image.open(imgpath).convert('L')
        name = imgpath.split('/')[-1]
        image_shape = np.array(image).shape
        image,label = self.transform_imglabel(image,label,name)
        # normalize image
        image = image / 255.0
        image -= 0.5
        image = image/0.5
        image = np.expand_dims(image, axis=0)
        image = torch.FloatTensor(image)
        aff = self.binary_relation_map(image) # [8, 128, 128]
        return {'img': image, 'aff':aff, 'path': imgpath,'shape':image_shape}

    def transform_imglabel(self,image,label,name):
        size = np.array(image, dtype=np.uint8).shape
        # print(name)
        name1 = name
        h,w = 2*size[0],2*size[1]
        if name1.startswith('site1') or name1.startswith('site2'):
            image = image.resize((w,h), Image.BICUBIC)
            label = label.resize((w,h), Image.NEAREST)
            image = np.asarray(image, np.float32)
            # label = self.encode_segmap(np.array(label, dtype=np.uint8))
        elif name1.startswith('site4'):
            image = image.resize((math.ceil(1.16*size[1]),math.ceil(1.16*size[0])), Image.BICUBIC)
            label = label.resize((math.ceil(1.16*size[1]),math.ceil(1.16*size[0])), Image.NEAREST)
            image = np.asarray(image, np.float32)
            # label = self.encode_segmap(np.array(label, dtype=np.uint8))
        elif name1.startswith('site3'):
            image = np.array(image, dtype=np.float32)
            # label = self.encode_segmap(np.array(label, dtype=np.uint8))

        image = np.expand_dims(image, axis=2)
        label = np.expand_dims(label, axis=2) 
        img_label = np.concatenate((image, label), axis=-1)
        img_label = self.CenterCrop(img_label,128)
        image = img_label[..., 0]
        label = img_label[..., 1]
        label = np.squeeze(label)
        image = np.squeeze(image)
        return image,label


    def binary_relation_map(self,image):
        images_from = np.array(image,dtype=np.float32)[0]
        # images_pad= np.zeros((images_from.shape[0]+2,images_from.shape[1]+2))
        # images_pad[1:-1,1:-1] = images_from
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

    def __len__(self):
        return len(self.imgpaths)

    def CenterCrop(self, imgarr, size):
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

    def name(self):
        return 'ResizeCganDataset'