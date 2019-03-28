import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import math 

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
        self.imgpaths = make_dataset(self.imgdir)
        self.imgpaths = sorted(self.imgpaths)

    def __getitem__(self, index):
        imgpath = self.imgpaths[index]
        name = imgpath.split('/')[-1]
        maskpath = os.path.join(self.maskdir,name)
        mask = Image.open(maskpath).convert('L')
        mask = self.imgtransform_mask(mask,name)
        imgL = Image.open(imgpath).convert('L')
        imgRGB = Image.open(imgpath).convert('RGB')
        imgL, image_shape = self.imgtransform_L(imgL,name)
        imgRGB, image_shape = self.imgtransform_RGB(imgRGB,name)
        # aff = self.binary_relation_map(imgL)
        aff = self.compute_relation_map(imgL)
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

    def imgtransform_L(self, image, name):
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
        mean = image.mean()
        std = image.std()
        pad_weight1, pad_weight2, pad_hight1, pad_hight2 = self.get_padshape(image)
        image = np.pad(image,((pad_hight1, pad_hight2),(pad_weight1, pad_weight2)),"constant")
        # normalize image 
        # image = anisodiff(image, 4, 0.1, 0.1)
        image = (image - mean)/(std+1e-10)
        image = (image - image.min()) / (image.max()-image.min()+1e-10) * 2.0
        image = image - 1.0
        # normalize image      
        image_shape = image.shape
        image = np.expand_dims(image, axis=0)
        image = torch.FloatTensor(image)
        return image, image_shape

    def imgtransform_mask(self, label, name):
        size = np.array(label, dtype=np.uint8).shape
        if name.startswith('site1') or name.startswith('site2'):
            label = label.resize((math.ceil(2.0*size[1]),math.ceil(2.0*size[0])), Image.NEAREST)
            label = np.array(label, dtype=np.uint8)
        elif name.startswith('site4'):
            label = label.resize((math.ceil(1.16*size[1]),math.ceil(1.16*size[0])), Image.NEAREST)
            label = np.array(label, dtype=np.uint8)
        elif name.startswith('site3'):
            label = label.resize((math.ceil(1.0*size[1]),math.ceil(1.0*size[0])), Image.NEAREST)
            label = np.array(label, dtype=np.uint8)
        pad_weight1, pad_weight2, pad_hight1, pad_hight2 = self.get_padshape(label)
        label = np.pad(label,((pad_hight1, pad_hight2),(pad_weight1, pad_weight2)),"constant")
        # label = np.expand_dims(label, axis=0)
        label = label / 255.0
        label = (label - label.min()) / (label.max()-label.min()) * 2.0
        label = label - 1.0
        label = np.expand_dims(label, axis=0)
        label = torch.FloatTensor(label)
        return label


    def imgtransform_RGB(self, image, name):
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


        pad_weight1, pad_weight2, pad_hight1, pad_hight2 = self.get_padshape(image)
        image = np.pad(image,((pad_hight1, pad_hight2),(pad_weight1, pad_weight2),(0,0)),"constant")
        # image = anisodiff(image, 3, 0.1, 0.1)
        # normalize image
        image = image / 255.0
        image = (image - np.mean(image))/(np.std(image)+1e-10)
        # image = (image - np.mean(image))/(np.std(image)+1e-10)   
        image = (image - image.min()) / (image.max()-image.min()+1e-10) * 2.0
        image = image - 1.0
        image_shape = image.shape
        image = image.transpose((2,0,1))
        # image = np.expand_dims(image, axis=0)
        image = torch.FloatTensor(image)
        return image, image_shape



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
        # print(relation_map.max(), relation_map.min())
        # relation_map = diff_maps.copy()
        return relation_map



    def __len__(self):
        return len(self.imgpaths)

    def name(self):
        return 'CganDataset'


