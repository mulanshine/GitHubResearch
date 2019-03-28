import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

# use crop label*image rec pad --> (32,24)or(64,48),resize to (64,48) for generator
class CganDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.phase = opt.phase
        self.imgdir = os.path.join(opt.dataroot,'cropimage_rect')
        self.imgpaths = make_dataset(self.imgdir)
        self.imgpaths = sorted(self.imgpaths)

    def __getitem__(self, index):
        imgpath = self.imgpaths[index]
        name = imgpath.split('/')[-1]
        img = Image.open(imgpath).convert('L')
        img,image_shape = self.imgtransform(img,name)
        aff = self.binary_relation_map(img)

        return {'img': img, 'aff':aff, 'path': imgpath,'shape':image_shape}

    def imgtransform(self,img,name):
        image = np.asarray(img, np.float32)
        if image.shape[0] <= 64 and image.shape[1] <= 48:
            if image.shape[0] > 32 or image.shape[1] > 24:
                hight = 64
                weight = 48
            else:
                hight = 32
                weight = 24
        else:
            print("#######################>64or>48#######################################")
            print(image.shape[0],image.shape[1])
            hight = image.shape[0]
            weight = image.shape[1]

        pad_weight1 = int((weight - image.shape[1])/2)
        pad_weight2 = weight - image.shape[1] - pad_weight1
        pad_hight1 = int((hight - image.shape[0])/2)
        pad_hight2 = hight - image.shape[0] - pad_hight1
        
        # if name.startswith('site1'):
        #     image -= 0.4767
        #     image = image/0.1168
        # elif name.startswith('site2'):
        #     image -= 0.5173
        #     image = image/0.1219
        # elif name.startswith('site3'):
        #     image -= 0.3329
        #     image = image/0.0551
        # elif name.startswith('site4'):
        #     image -= 0.5060
        #     image = image/0.1311

        image = np.pad(image,((pad_hight1, pad_hight2),(pad_weight1, pad_weight2)),"constant")        
        image = image / 255.0
        image -= 0.5
        image = image/0.5
        if image.shape != (64,48):
            image = Image.fromarray(image)
            image = image.resize((48,64)) # note (48,64) not (64,48)
            image = np.asarray(image, np.float32)
        image_shape = image.shape
        image = np.expand_dims(image, axis=0)
        image = torch.FloatTensor(image)
        return image,image_shape

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

    def __len__(self):
        return len(self.imgpaths)

    def name(self):
        return 'CganDataset'


