import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import math 

# use(80,48),resize to (80,48) for generator
class CganDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.phase = opt.phase
        self.imgdir = os.path.join(opt.dataroot,'cropimage_rect')
        self.maskdir = os.path.join(opt.dataroot,'cropmask_rect')
        self.imgpaths = make_dataset(self.imgdir,opt.site)
        self.imgpaths = sorted(self.imgpaths)

    def __getitem__(self, index):
        imgpath = self.imgpaths[index]
        name = imgpath.split('/')[-1]
        # maskpath = os.path.join(self.maskdir,name)
        # mask = Image.open(maskpath).convert('L')
        # mask = self.imgtransform_mask(mask,name)
        imgL = Image.open(imgpath).convert('L')
        # imgRGB = Image.open(imgpath).convert('RGB')
        imgL, imgRGB, image_shape = self.imgtransform(imgL,name)
        # imgRGB, image_shape = self.imgtransform_RGB(imgRGB,name)
        aff = self.binary_relation_map(imgL)
        # aff = self.compute_relation_map(imgL)
        return {'img': imgRGB, 'aff':aff, 'path': imgpath,'shape':image_shape}

        
    def get_padshape(self, image):
        if image.shape[0] <= 80 and image.shape[1] <= 64:
            hight = 80
            weight = 64
            pad_weight1 = int((weight - image.shape[1])/2)
            pad_weight2 = weight - image.shape[1] - pad_weight1
            pad_hight1 = int((hight - image.shape[0])/2)
            pad_hight2 = hight - image.shape[0] - pad_hight1
        elif image.shape[0] > 80 and image.shape[1] <= 64:
            print("#######################>80or<48#######################################")
            print(image.shape[0],image.shape[1])
            weight = 64
            pad_weight1 = int((weight - image.shape[1])/2)
            pad_weight2 = weight - image.shape[1] - pad_weight1
            pad_hight1 = 0
            pad_hight2 = 0
        elif image.shape[0] < 80 and image.shape[1] > 64:
            print("#######################<80or>48#######################################")
            print(image.shape[0],image.shape[1])
            hight = 80
            pad_weight1 = 0
            pad_weight2 = 0
            pad_hight1 = int((hight - image.shape[0])/2)
            pad_hight2 = hight - image.shape[0] - pad_hight1
        elif image.shape[0] > 80 and image.shape[1] > 64:
            print("#######################>80or>48#######################################")
            print(image.shape[0],image.shape[1])
            pad_weight1 = 0
            pad_weight2 = 0
            pad_hight1 = 0
            pad_hight2 = 0
        return pad_weight1, pad_weight2, pad_hight1, pad_hight2

    def imgtransform(self, image, name):
        size = np.array(image, dtype=np.uint8).shape
        if name.startswith('site1') or name.startswith('site2'):
            image = image.resize((math.ceil(2*size[1]),math.ceil(2*size[0])), Image.BICUBIC)
            image = np.asarray(image, np.float32)
        elif name.startswith('site4'):
            image = image.resize((math.ceil(1.16*size[1]),math.ceil(1.16*size[0])), Image.BICUBIC)
            image = np.asarray(image, np.float32)
        elif name.startswith('site3'):
            image = image.resize((math.ceil(1.0*size[1]),math.ceil(1.0*size[0])), Image.BICUBIC)
            image = np.array(image, dtype=np.float32)

        image = image / 255.0
        pad_weight1, pad_weight2, pad_hight1, pad_hight2 = self.get_padshape(image)
        image = np.pad(image,((pad_hight1, pad_hight2),(pad_weight1, pad_weight2)),"constant")
        # normalize image 
        # image = anisodiff(image, 4, 0.1, 0.1)
        image = (image - 0.5) * 2.0
        # normalize image      
        image_shape = image.shape
        image = np.expand_dims(image, axis=0)
        imgRGB = np.concatenate((image,image,image),axis=0)
        image = torch.FloatTensor(image)
        imgRGB = torch.FloatTensor(imgRGB)
        return image, imgRGB, image_shape


    def binary_relation_map(self,image):
        images_from = np.array(image,dtype=np.float32)[0]
        images_pad = np.pad(images_from,((3,3),(3,3)),'constant')
        images_to = np.zeros((8,images_from.shape[0],images_from.shape[1])) #,constant_values = (0.0,0.0)
        # images_to[0] = images_pad[:-2, 2:]
        # images_to[1] = images_pad[1:-1,2:]
        # images_to[2] = images_pad[2:, 2:]
        # images_to[3] = images_pad[:-2,1:-1]
        # images_to[4] = images_pad[2:,1:-1]
        # images_to[5] = images_pad[:-2,:-2]
        # images_to[6] = images_pad[1:-1,:-2]
        # images_to[7] = images_pad[2:, :-2]

        # 24 channels
        # images_to[0] = images_pad[:-4, 4:]
        # images_to[1] = images_pad[1:-3,4:]
        # images_to[2] = images_pad[2:-2,4:]
        # images_to[3] = images_pad[3:-1,4:]
        # images_to[4] = images_pad[4:,4:]

        # images_to[5] = images_pad[:-4, 3:-1]
        # images_to[6] = images_pad[1:-3,3:-1]
        # images_to[7] = images_pad[2:-2,3:-1]
        # images_to[8] = images_pad[3:-1,3:-1]
        # images_to[9] = images_pad[4:,3:-1]

        # images_to[10] = images_pad[:-4,2:-2]
        # images_to[11] = images_pad[1:-3,2:-2]
        # images_to[12] = images_pad[3:-1,2:-2]
        # images_to[13] = images_pad[4:,2:-2]

        # images_to[14] = images_pad[:-4,1:-3]
        # images_to[15] = images_pad[1:-3,1:-3]
        # images_to[16] = images_pad[2:-2,1:-3]
        # images_to[17] = images_pad[3:-1,1:-3]
        # images_to[18] = images_pad[4:,1:-3]

        # images_to[19] = images_pad[:-4,:-4]
        # images_to[20] = images_pad[1:-3,:-4]
        # images_to[21] = images_pad[2:-2,:-4]
        # images_to[22] = images_pad[3:-1,:-4]
        # images_to[23] = images_pad[4:,1:-3]

        # # 8 channels_interval_1
        # images_to[0] = images_pad[:-4,:-4]
        # images_to[1] = images_pad[:-4,2:-2]
        # images_to[2] = images_pad[:-4,4:]

        # images_to[3] = images_pad[2:-2,:-4]
        # images_to[4] = images_pad[2:-2,4:]

        # images_to[5] = images_pad[4:,:-4]
        # images_to[6] = images_pad[4:,2:-2]
        # images_to[7] = images_pad[4:,4:]

        # 8 channels_interval_2
        images_to[0] = images_pad[:-6,:-6]
        images_to[1] = images_pad[:-6,3:-3]
        images_to[2] = images_pad[:-6,6:]

        images_to[3] = images_pad[3:-3,:-6]
        images_to[4] = images_pad[3:-3,6:]

        images_to[5] = images_pad[6:,:-6]
        images_to[6] = images_pad[6:,3:-3]
        images_to[7] = images_pad[6:,6:]


        diff_maps = images_to - images_from
        diff_maps[diff_maps>=0] = 1.0
        # diff_maps[diff_maps<0] = -1.0
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
        # print(relation_map.max(), relation_map.min())
        # relation_map = diff_maps.copy()
        return relation_map

    def __len__(self):
        return len(self.imgpaths)

    def name(self):
        return 'CganDataset'


