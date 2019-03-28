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

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.pgm','.PGM'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


#root: /media/jjchu/DataSets/spinalcord/train/cropmask|cropimage
class spinalcordCropDataSet(data.Dataset):
    def __init__(self, root, img_size=(80,48), site=['site2','site4'], batchsize=1, n_class=3,nlabel=True,set='train'):
        self.root = root
        self.set = set
        self.site = site
        self.imgdir = os.path.join(self.root,'train/cropimage_rect')
        self.maskdir = os.path.join(self.root,'train/cropmask_rect')
        self.names = self.make_imagenames(self.imgdir,self.site)
        self.imgnames = sorted(self.names)
        self.batchsize = batchsize
        self.n_class = n_class
        self.img_size = img_size
        self.valid_classes = [0, 128, 255]
        if self.n_class == 3:
            self.train_classes = [0, 1, 2]
        elif self.n_class == 2:
            self.train_classes = [0, 1, 1]

        self.class_map = dict(zip(self.valid_classes, self.train_classes))
        self.files = []
        self.nlabel = nlabel
        
        for img_name in self.imgnames: # cropmask|cropimage
            img_file = osp.join(self.imgdir, img_name)
            lbl_file = osp.join(self.maskdir, img_name)
            self.files.append({
                "img": img_file,
                "lbl": lbl_file,
                "name": img_name
            }) 

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('L')
        name = datafiles["name"]
        label = Image.open(datafiles["lbl"]).convert('L')
        size = np.array(label).shape
        image,label = self.imgtransform(image, label,name) 
        if self.nlabel:
            lbl = np.zeros((self.n_class,label.shape[1],label.shape[2]))
            for i in range(self.n_class):
                lbl[i] = np.array((label == i), dtype=np.uint8)
        else:
            lbl = label
        return image, lbl, label, size, name

    def encode_segmap(self, mask):
        for validc in self.valid_classes:
            mask[mask==validc] = self.class_map[validc]
        return mask

    def get_padshape(self, image):
        if image.shape[0] <= 80 and image.shape[1] <= 48:
            hight = 80
            weight = 48
            pad_weight1 = int((weight - image.shape[1])/2)
            pad_weight2 = weight - image.shape[1] - pad_weight1
            pad_hight1 = int((hight - image.shape[0])/2)
            pad_hight2 = hight - image.shape[0] - pad_hight1
        else:
            print("#######################>80or>48#######################################")
            print(image.shape[0],image.shape[1])
            pad_weight1 = 0
            pad_weight2 = 0
            pad_hight1 = 0
            pad_hight2 = 0
        return pad_weight1, pad_weight2, pad_hight1, pad_hight2

    def imgtransform(self, image, label,name):
        size = np.array(image, dtype=np.uint8).shape
        name1 = name
        h,w = 2*size[0],2*size[1]
        if name1.startswith('site1') or name1.startswith('site2'):
            image = image.resize((math.ceil(2*size[1]),math.ceil(2*size[0])), Image.BICUBIC)
            label = label.resize((math.ceil(2*size[1]),math.ceil(2*size[0])), Image.NEAREST)
            image = np.asarray(image, np.float32)
            label = self.encode_segmap(np.array(label, dtype=np.uint8))
        elif name1.startswith('site4'):
            image = image.resize((math.ceil(1.16*size[1]),math.ceil(1.16*size[0])), Image.BICUBIC)
            label = label.resize((math.ceil(1.16*size[1]),math.ceil(1.16*size[0])), Image.NEAREST)
            image = np.asarray(image, np.float32)
            label = self.encode_segmap(np.array(label, dtype=np.uint8))
        elif name1.startswith('site3'):
            image = np.array(image, dtype=np.float32)
            label = self.encode_segmap(np.array(label, dtype=np.uint8))
        
        pad_weight1, pad_weight2, pad_hight1, pad_hight2 = self.get_padshape(image)
        image = np.expand_dims(image, axis=2) 
        label = np.expand_dims(label, axis=2) 
        img_label = np.concatenate((image, label), axis=-1)
        img_label = np.pad(img_label,((pad_hight1, pad_hight2),(pad_weight1, pad_weight2),(0,0)),"constant")
        image = img_label[..., 0]
        image = np.squeeze(image)
        label = img_label[..., 1]
        label = np.squeeze(label)
        # normalize image
        image = image / 255.0
        image -= np.mean(image)
        image = image/np.std(image)
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)
        image = torch.FloatTensor(image)
        label = torch.FloatTensor(label)
        return image,label

    def make_imagenames(self,dir,site):
        imagenames = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname): # use site1 and site3 for training,use site1 and site3 for testing,
                    if fname.startswith(site[0]) or fname.startswith(site[1]) :
                        imagenames.append(fname)
        return imagenames

    def name(self):
        return 'spinalcordCropDataSet'


class spinalcordRealCropDataSet(data.Dataset):
    def __init__(self, root, img_size=(100,100), site=['site1','site2'], batchsize=1, n_class=3,nlabel=True,set='train'):
        self.root = root
        self.set = set
        self.site = site
        self.imgdir = os.path.join(self.root,'crop_100/image_crop/') # 
        self.maskdir = os.path.join(self.root,'crop_100/mask_crop/') # "/home/jjchu/spinalcord/"
        self.names = self.make_imagenames(self.imgdir,self.site)
        self.imgnames = sorted(self.names)
        self.batchsize = batchsize
        self.n_class = n_class
        self.img_size = img_size
        self.valid_classes = [0, 128, 255]
        if self.n_class == 3:
            self.train_classes = [0, 1, 2]
        elif self.n_class == 2:
            self.train_classes = [0, 1, 0]

        self.class_map = dict(zip(self.valid_classes, self.train_classes))
        self.files = []
        self.nlabel = nlabel
        
        for img_name in self.imgnames: # cropmask|cropimage
            img_file = osp.join(self.imgdir, img_name)
            lbl_file = osp.join(self.maskdir, img_name)
            self.files.append({
                "img": img_file,
                "lbl": lbl_file,
                "name": img_name
            }) 

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]
        label = Image.open(datafiles["lbl"]).convert('L')
        size = np.array(label).shape
        image,label = self.imgtransform(image, label) 
        if self.nlabel:
            lbl = np.zeros((self.n_class,label.shape[1],label.shape[2]))
            for i in range(self.n_class):
                lbl[i] = np.array((label == i), dtype=np.uint8)
        else:
            lbl = label
        return image, lbl, label, size, name

    def encode_segmap(self, mask):
        for validc in self.valid_classes:
            mask[mask==validc] = self.class_map[validc]
        return mask

    def imgtransform(self, image, label):
        image = np.asarray(image, np.float32)
        label = self.encode_segmap(np.array(label, dtype=np.uint8))
        # normalize image
        image = image / 255.0
        image -= np.mean(image)
        image = image/np.std(image)
        # image -= 0.5
        # image = image/0.5
        image = np.transpose(image,(2,0,1))
        label = np.expand_dims(label, axis=0)
        image = torch.FloatTensor(image)
        label = torch.FloatTensor(label)
        return image,label

    def make_imagenames(self,dir,site):
        imagenames = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname): # use site1 and site3 for training,use site1 and site3 for testing,
                    if fname.startswith(site[0]) or fname.startswith(site[1]) :
                        imagenames.append(fname)
        return imagenames

    def name(self):
        return 'spinalcordRealCropDataSet'


class spinalcordresGen8064DataSet(data.Dataset):
    def __init__(self, root, img_size=(80,64), site=['site1','site2'], batchsize=1, n_class=3,nlabel=True,set='train',real_or_fake='fake'):
        self.root = root
        self.set = set
        self.imgdir = self.root 
        self.maskdir = "/home/jjchu/DataSet/spinalcord/crop_8064/mask_crop/"
        # self.maskdir = "/home/jjchu/DataSet/spinalcord/crop_100/mask_crop/"
        # self.maskdir = "/home/jjchu/DataSet/spinalcord/cropmask_rect/"
        self.site = site # ['site2','site4']
        self.real_or_fake = real_or_fake
        if self.real_or_fake =='fake' or self.real_or_fake == 'real':
            self.endstr ="_"+self.real_or_fake+'.pgm'
        elif self.real_or_fake == 'real_fake':
            self.endstr = '.pgm'

        self.names = self.make_imagenames(self.imgdir,self.site)
        self.imgnames = sorted(self.names)
        self.batchsize = batchsize
        self.n_class = n_class
        self.img_size = img_size
        self.valid_classes = [0, 128, 255]
        self.train_classes = [0, 1, 2]
        self.class_map = dict(zip(self.valid_classes, self.train_classes))
        self.files = []
        self.nlabel = nlabel
        
        for img_name in self.imgnames: # cropmask|cropimage
            mask_name = '_'.join(img_name.split('_')[:-1])+'.pgm'
            img_file = osp.join(self.imgdir, img_name)
            lbl_file = osp.join(self.maskdir, mask_name)
            self.files.append({
                "img": img_file,
                "lbl": lbl_file,
                "name": img_name
            })        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]
        label = Image.open(datafiles["lbl"]).convert('L')
        image,label = self.imgtransform(image, label, name)
        size = label.shape        
        if self.nlabel:
            lbl = np.zeros((self.n_class,label.shape[1],label.shape[2]))
            for i in range(self.n_class):
                lbl[i] = np.array((label == i), dtype=np.uint8)
        else:
            lbl = label
        return image, lbl, label, size, name

    def encode_segmap(self, mask):
        for validc in self.valid_classes:
            mask[mask==validc] = self.class_map[validc]
        return mask

    def imgtransform(self, image, label,name):
        size = np.array(label, dtype=np.uint8).shape
        name1 = name
        label = self.encode_segmap(np.array(label, dtype=np.uint8))
        # normalize label
        image = np.array(image,dtype=np.float32)
        image = image / 255.0
        # image -= np.mean(image)
        # image = image/np.std(image)
        image -= 0.5
        image = image/0.5
        image = np.transpose(image,(2,0,1))
        label = np.expand_dims(label, axis=0)
        image = torch.FloatTensor(image)
        label = torch.FloatTensor(label)
        return image, label

    def make_imagenames(self,dir,site):
        endstr = self.endstr
        imagenames = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname): # use site1 and site3 for training,use site1 and site3 for testing,
                    if fname.endswith(endstr) and not fname.endswith('_target.pgm'):
                        if fname.startswith(site[0]) or fname.startswith(site[1]) :
                            imagenames.append(fname)
                            # if fname.endswith('_fake.pgm'):
                            #     imagenames.append(fname)
                            # elif fname.endswith('_real.pgm'):
                            #     if random.random()>0.5:
                            #         imagenames.append(fname)
        return imagenames

    def name(self):
        return 'spinalcordresGen8064DataSet'


class spinalcordGen100DataSet(data.Dataset):
    def __init__(self, root, img_size=(128,128), site=['site1','site2'], batchsize=1, n_class=3, nlabel=True, set='train',real_or_fake='real'):
        self.root = root
        self.set = set
        self.imgdir = self.root 
        self.maskdir = "/home/jjchu/DataSet/spinalcord/crop_100/mask_crop/"
        self.real_or_fake = real_or_fake
        if self.real_or_fake =='fake' or self.real_or_fake == 'real':
            self.endstr ="_" + self.real_or_fake+'.pgm'
        elif self.real_or_fake == 'real_fake':
            self.endstr = '.pgm'

        self.site = site # ['site2','site4']
        self.names = self.make_imagenames(self.imgdir,self.site)
        self.imgnames = sorted(self.names)

        self.batchsize = batchsize
        self.n_class = n_class
        self.img_size = img_size
        self.valid_classes = [0, 128, 255]
        if self.n_class == 3:
            self.train_classes = [0, 1, 2]
        elif self.n_class == 2:
            self.train_classes = [0, 1, 0]
        self.class_map = dict(zip(self.valid_classes, self.train_classes))
        self.files = []
        self.nlabel = nlabel
        
        
        for img_name in self.imgnames: # cropmask|cropimage
            mask_name = '_'.join(img_name.split('_')[:-1])+'.pgm'
            img_file = osp.join(self.imgdir, img_name)
            lbl_file = osp.join(self.maskdir, mask_name)
            self.files.append({
                "img": img_file,
                "lbl": lbl_file,
                "name": img_name
            })        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]
        label = Image.open(datafiles["lbl"])
        image,label = self.imgtransform(image, label)
        size = label.shape
        image = np.transpose(image,(2,0,1))
        # image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)
        image = torch.FloatTensor(image)
        label = torch.FloatTensor(label)
        
        if self.nlabel:
            lbl = np.zeros((self.n_class,label.shape[1],label.shape[2]))
            for i in range(self.n_class):
                lbl[i] = np.array((label == i), dtype=np.uint8)
        else:
            lbl = label
        return image, lbl, label, size, name

    def encode_segmap(self, mask):
        for validc in self.valid_classes:
            mask[mask==validc] = self.class_map[validc]
        return mask


    def imgtransform(self, img, label):
        image = np.array(img, dtype=np.float32)
        label = np.array(label, dtype=np.uint8)
        label = self.encode_segmap(label)
        # normalize image
        image = image / 255.0
        image -= np.mean(image)
        image = image/np.std(image)
        image -= 0.5
        image = image/0.5
        return image,label

    def make_imagenames(self,dir,site):
        imagenames = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname): # use site1 and site3 for training,use site1 and site3 for testing,
                    if fname.endswith(self.endstr) and not fname.endswith('_target.pgm'):
                        if fname.startswith(site[0]) or fname.startswith(site[1]) :
                            imagenames.append(fname)
        return imagenames


    def name(self):
        return 'spinalcordGen128DataSet'



# img_size=(80,48) resize2pad
class spinalcordGenlblresize2padataSet(data.Dataset):
    def __init__(self, root, img_size=(80,64), site=['site1','site2'], batchsize=1, n_class=3,nlabel=True,set='train',real_or_fake='fake'):
        self.root = root
        self.set = set
        self.imgdir = self.root 
        self.maskdir = "/home/jjchu/DataSet/spinalcord/cropmask_rect/"
        self.site = site # ['site2','site4']
        self.real_or_fake = real_or_fake
        if self.real_or_fake =='fake' or self.real_or_fake == 'real':
            self.endstr ="_"+self.real_or_fake+'.pgm'
        elif self.real_or_fake == 'real_fake':
            self.endstr = '.pgm'
        self.names = self.make_imagenames(self.imgdir,self.site)
        self.imgnames = sorted(self.names)
        self.batchsize = batchsize
        self.n_class = n_class
        self.img_size = img_size
        self.valid_classes = [0, 128, 255]
        self.train_classes = [0, 1, 2]
        self.class_map = dict(zip(self.valid_classes, self.train_classes))
        self.files = []
        self.nlabel = nlabel
        
        for img_name in self.imgnames: # cropmask|cropimage
            mask_name = '_'.join(img_name.split('_')[:-1])+'.pgm'
            img_file = osp.join(self.imgdir, img_name)
            lbl_file = osp.join(self.maskdir, mask_name)
            self.files.append({
                "img": img_file,
                "lbl": lbl_file,
                "name": img_name
            })        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]
        label = Image.open(datafiles["lbl"]).convert('L')
        image,label = self.imgtransform(image, label, name)
        size = label.shape        
        if self.nlabel:
            lbl = np.zeros((self.n_class,label.shape[1],label.shape[2]))
            for i in range(self.n_class):
                lbl[i] = np.array((label == i), dtype=np.uint8)
        else:
            lbl = label
        return image, lbl, label, size, name

    def encode_segmap(self, mask):
        for validc in self.valid_classes:
            mask[mask==validc] = self.class_map[validc]
        return mask

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

    def imgtransform(self, image, label,name):
        size = np.array(label, dtype=np.uint8).shape
        name1 = name
        if name1.startswith('site1') or name1.startswith('site2'):
            label = label.resize((math.ceil(2*size[1]),math.ceil(2*size[0])), Image.NEAREST)
            label = self.encode_segmap(np.array(label, dtype=np.uint8))
        elif name1.startswith('site4'):
            label = label.resize((math.ceil(1.16*size[1]),math.ceil(1.16*size[0])), Image.NEAREST)
            label = self.encode_segmap(np.array(label, dtype=np.uint8))
        elif name1.startswith('site3'):
            label = self.encode_segmap(np.array(label, dtype=np.uint8))

        pad_weight1, pad_weight2, pad_hight1, pad_hight2 = self.get_padshape(label)
        label = self.encode_segmap(np.array(label, dtype=np.uint8))
        label = np.pad(label,((pad_hight1, pad_hight2),(pad_weight1, pad_weight2)),"constant")
        # normalize label
        image = np.array(image,dtype=np.float32)
        image = image / 255.0
        # image -= np.mean(image)
        # image = image/np.std(image)
        image -= 0.5
        image = image/0.5
        # image_shape = image.shape
        image = np.transpose(image,(2,0,1))
        label = np.expand_dims(label, axis=0)
        image = torch.FloatTensor(image)
        label = torch.FloatTensor(label)
        return image, label

    def make_imagenames(self,dir,site):
        imagenames = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname): # use site1 and site3 for training,use site1 and site3 for testing,
                    if fname.endswith(self.endstr):
                        if fname.startswith(site[0]) or fname.startswith(site[1]) :
                            imagenames.append(fname)
        return imagenames

    def name(self):
        return 'spinalcordGenDataSet'

class spinalcordGenlblpad2resizeDataSet(data.Dataset):
    def __init__(self, root, img_size=(80,64), site=['site1','site2'], batchsize=1, n_class=3,nlabel=True,set='train'):
        self.root = root
        self.set = set
        self.imgdir = self.root 
        self.maskdir = "/home/jjchu/DataSet/spinalcord/cropmask_rect/"
        self.site = site # ['site2','site4']
        self.names = self.make_imagenames(self.imgdir,self.site)
        self.imgnames = sorted(self.names)

        self.batchsize = batchsize
        self.n_class = n_class
        self.img_size = img_size
        self.valid_classes = [0, 128, 255]
        self.train_classes = [0, 1, 2]
        self.class_map = dict(zip(self.valid_classes, self.train_classes))
        self.files = []
        self.nlabel = nlabel
        
        
        for img_name in self.imgnames: # cropmask|cropimage

            mask_name = '_'.join(img_name.split('_')[:-1])+'.pgm'

            img_file = osp.join(self.imgdir, img_name)
            lbl_file = osp.join(self.maskdir, mask_name)
            self.files.append({
                "img": img_file,
                "lbl": lbl_file,
                "name": img_name
            })        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('L')
        name = datafiles["name"]
        label = Image.open(datafiles["lbl"])
        image,label = self.imgtransform(image, label)
        size = label.shape
        label = self.image_resize(image,label)
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)
        # print(image.shape)
        # print(label.shape)
        image = torch.FloatTensor(image)
        label = torch.FloatTensor(label)
        
        if self.nlabel:
            lbl = np.zeros((self.n_class,label.shape[1],label.shape[2]))
            for i in range(self.n_class):
                lbl[i] = np.array((label == i), dtype=np.uint8)
        else:
            lbl = label
        return image, lbl, label, size, name

    def encode_segmap(self, mask):
        for validc in self.valid_classes:
            mask[mask==validc] = self.class_map[validc]
        return mask

    def get_padshape(self, image):
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
            hight = 64
            weight = 48
   
        pad_weight1 = int((weight - image.shape[1])/2)
        pad_weight2 = weight - image.shape[1] - pad_weight1
        pad_hight1 = int((hight - image.shape[0])/2)
        pad_hight2 = hight - image.shape[0] - pad_hight1
        return pad_weight1, pad_weight2, pad_hight1, pad_hight2

    def imgtransform(self, img, label):
        image = np.array(img, dtype=np.float32)
        label = np.array(label, dtype=np.uint8)
        label = self.encode_segmap(label)
        pad_weight1, pad_weight2, pad_hight1, pad_hight2 = self.get_padshape(label)
        label = np.pad(label,((pad_hight1, pad_hight2),(pad_weight1, pad_weight2)),"constant")
        # normalize image
        image = image / 255.0
        image -= np.mean(image)
        image = image/np.std(image)
        return image,label

    def image_resize(self,image,label):
        if label.shape != (64,48):
            label = Image.fromarray(label)
            label = label.resize((48,64)) # note (48,64) not (64,48)
            label = np.asarray(label, np.uint8)
        return label

    def make_imagenames(self,dir,site):
        imagenames = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname): # use site1 and site3 for training,use site1 and site3 for testing,
                    if fname.endswith('_fake.pgm'):
                        if fname.startswith(site[0]) or fname.startswith(site[1]) :
                            imagenames.append(fname)
        return imagenames

    def name(self):
        return 'spinalcordGenDataSet'


class spinalcordGenNoLblResizeDataSet(data.Dataset):
    def __init__(self, root, img_size=(64,48), batchsize=1, n_class=3,nlabel=True,set='train'):
        self.root = root
        self.set = set
        self.imgdir = self.root 
        self.maskdir = "/media/jjchu/DataSets/spinalcord/train/cropmask/"
        self.names = self.make_imagenames(self.imgdir)
        self.imgnames = sorted(self.names)

        self.batchsize = batchsize
        self.n_class = n_class
        self.img_size = img_size
        self.valid_classes = [0, 128, 255]
        self.train_classes = [0, 1, 2]
        self.class_map = dict(zip(self.valid_classes, self.train_classes))
        self.files = []
        self.nlabel = nlabel
        
        for img_name in self.imgnames: # cropmask|cropimage

            mask_name = '_'.join(img_name.split('_')[:-1])+'.pgm'
            img_file = osp.join(self.imgdir, img_name)
            lbl_file = osp.join(self.maskdir, mask_name)
            self.files.append({
                "img": img_file,
                "lbl": lbl_file,
                "name": img_name
            })        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('L')
        name = datafiles["name"]
        label = Image.open(datafiles["lbl"])
        image,label = self.imgtransform(image, label)

        # label = self.image_resize(image,label)

        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)
        image = torch.FloatTensor(image)
        label = torch.FloatTensor(label)
        
        if self.nlabel:
            size = label.shape
            lbl = np.zeros((self.n_class,label.shape[1],label.shape[2]))
            for i in range(self.n_class):
                lbl[i] = np.array((label == i), dtype=np.uint8)
        else:
            lbl = label
        return image, lbl, label, size, name

    def encode_segmap(self, mask):
        for validc in self.valid_classes:
            mask[mask==validc] = self.class_map[validc]
        return mask

    def get_padshape(self, image):
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
            hight = 64
            weight = 48
   
        pad_weight1 = int((weight - image.shape[1])/2)
        pad_weight2 = weight - image.shape[1] - pad_weight1
        pad_hight1 = int((hight - image.shape[0])/2)
        pad_hight2 = hight - image.shape[0] - pad_hight1
        return pad_weight1, pad_weight2, pad_hight1, pad_hight2

    def imgtransform(self, img, label):
        image = np.array(img, dtype=np.float32)
        label = np.array(label, dtype=np.uint8)
        label = self.encode_segmap(label)
        pad_weight1, pad_weight2, pad_hight1, pad_hight2 = self.get_padshape(label)
        label = np.pad(label,((pad_hight1, pad_hight2),(pad_weight1, pad_weight2)),"constant")
        # normalize image
        image = image / 255.0
        image -= 0.5
        image = image/0.5
        return image,label

    # def image_resize(self,image,label):
    #     if label.shape != image.shape:
    #         label = Image.fromarray(label)
    #         label = label.resize(image.shape) # note (48,64) not (64,48)
    #         label = np.asarray(label, np.uint8)
    #     return label

    def make_imagenames(self,dir):
        imagenames = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname): # use site1 and site3 for training,use site1 and site3 for testing,
                    if fname.endswith('_fake.pgm'):
                        if fname.startswith('site2') or fname.startswith('site4') :
                            imagenames.append(fname)
        return imagenames

    def name(self):
        return 'spinalcordGenNoLblResizeDataSet'



class spinalcordGenlblresize2padatatestSet(data.Dataset):
    def __init__(self, root, list_path, img_size=(80,64), site=['site1','site2'], batchsize=1, n_class=3,nlabel=True,set='train'):
        self.root = root
        self.list_path = list_path
        self.set = set
        self.imgdir = self.root 
        self.maskdir = "/media/jjchu/DataSets/spinalcord/train/cropmask_rect/"
        self.site = site # ['site2','site4']
        self.names = self.make_imagenames(self.imgdir,self.site)
        self.imgnames = sorted(self.names)
        self.batchsize = batchsize
        self.n_class = n_class
        self.img_size = img_size
        self.valid_classes = [0, 128, 255]
        self.train_classes = [0, 1, 2]
        self.class_map = dict(zip(self.valid_classes, self.train_classes))
        self.files = []
        self.nlabel = nlabel
        self.img_masks = [name.strip().split(' ') for name in open(list_path)]

        for img_mask in self.img_masks:
            img_file = osp.join(self.imgdir, img_mask[0])
            lbl_file = osp.join(self.maskdir, img_mask[1])
            self.files.append({
                "img": img_file,
                "lbl": lbl_file,
                "name": img_mask   
            })        


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('L')
        name = datafiles["name"]
        label = Image.open(datafiles["lbl"]).convert('L')
        image,label = self.imgtransform(image, label, name)
        size = label.shape        
        if self.nlabel:
            lbl = np.zeros((self.n_class,label.shape[1],label.shape[2]))
            for i in range(self.n_class):
                lbl[i] = np.array((label == i), dtype=np.uint8)
        else:
            lbl = label
        return image, lbl, label, size, name

    def encode_segmap(self, mask):
        for validc in self.valid_classes:
            mask[mask==validc] = self.class_map[validc]
        return mask

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

    def imgtransform(self, image, label,name):
        size = np.array(label, dtype=np.uint8).shape
        name1 = name
        if name1.startswith('site1') or name1.startswith('site2'):
            label = label.resize((math.ceil(2*size[1]),math.ceil(2*size[0])), Image.NEAREST)
            label = self.encode_segmap(np.array(label, dtype=np.uint8))
        elif name1.startswith('site4'):
            label = label.resize((math.ceil(1.16*size[1]),math.ceil(1.16*size[0])), Image.NEAREST)
            label = self.encode_segmap(np.array(label, dtype=np.uint8))
        elif name1.startswith('site3'):
            label = self.encode_segmap(np.array(label, dtype=np.uint8))

        pad_weight1, pad_weight2, pad_hight1, pad_hight2 = self.get_padshape(label)
        label = self.encode_segmap(np.array(label, dtype=np.uint8))
        label = np.pad(label,((pad_hight1, pad_hight2),(pad_weight1, pad_weight2)),"constant")
        # normalize label
        image = np.array(image,dtype=np.float32)
        image = image / 255.0
        image -= np.mean(image)
        image = image/np.std(image)
        # image -= 0.5
        # image = image/0.5
        # image_shape = image.shape
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)
        image = torch.FloatTensor(image)
        label = torch.FloatTensor(label)
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
        return image, label

    def make_imagenames(self,dir,site):
        imagenames = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname): # use site1 and site3 for training,use site1 and site3 for testing,
                    if fname.endswith('_fake.pgm'):
                        if fname.startswith(site[0]) or fname.startswith(site[1]) :
                            imagenames.append(fname)
        return imagenames

    def name(self):
        return 'spinalcordGenDataSet'

