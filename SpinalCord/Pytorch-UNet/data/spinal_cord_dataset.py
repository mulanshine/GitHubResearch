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

# train_pairs_list = "/home/jjchu/My_Research/spinal_cord_segmentation/data/train_site12_list.txt"
# site2-sc04-image_1.pgm site2-sc04-mask-r3_1.pgm
# root = "/public/jjchu/Dataset/Spinel_cord_segmentation/"train/|mask|images|
class spinalcordDataSet(data.Dataset):
    def __init__(self, root,list_path, max_iters=None, img_size=128, crop_size=128,resize_and_crop='resize', batchsize=1, n_class=2,nlabel=True,set='train'):
        self.root = root
        self.list_path = list_path
        self.set = set
        self.batchsize = batchsize
        self.resize_and_crop = resize_and_crop
        self.n_class = n_class
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.img_size = (img_size,img_size)
        self.crop_size = crop_size
        self.valid_classes = [0, 128, 255]
        self.train_classes = [0, 1, 1]
        self.class_map = dict(zip(self.valid_classes, self.train_classes))
        self.files = []
        self.nlabel = nlabel
        self.img_masks = [name.strip().split(' ') for name in open(list_path)]
        for img_mask in self.img_masks:
            # use train
            img_file = osp.join(self.root, "%s/images/%s" % ('train', img_mask[0]))
            # if self.set =='train':
            lbl_file = osp.join(self.root, "%s/mask/%s"%('train', img_mask[1]))
            # elif self.set =='test':
            #     lbl_file = osp.join(self.root, "%s/images/%s"%(self.set, img_mask[0]))
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
        if self.set =='test':
            if self.resize_and_crop == "center_crop":
                size = np.array(label, dtype=np.uint8).shape[1]
                if size <= self.crop_size:
                    image = np.asarray(image, np.float32)
                    label = self.encode_segmap(np.array(label, dtype=np.uint8))
                else:
                    image = np.asarray(image, np.float32)
                    label = self.encode_segmap(np.array(label, dtype=np.uint8))
                    image = np.expand_dims(image, axis=2) 
                    label = np.expand_dims(label, axis=2) 
                    img_label = np.concatenate((image, label), axis=-1)
                    img_label = CenterCrop(img_label,self.crop_size)
                    image = img_label[..., 0]
                    label = img_label[..., 1]
                    label = np.squeeze(label)
                    image = np.squeeze(image)

        if self.set =='train':
            if self.resize_and_crop == "resize":
                image = image.resize(self.img_size, Image.BICUBIC)
                image = np.asarray(image, np.float32)
                label = label.resize(self.img_size, Image.NEAREST)
                label = self.encode_segmap(np.array(label, dtype=np.uint8))

            elif self.resize_and_crop == "center_crop":
                size = np.array(label, dtype=np.uint8).shape
                if size[1] <= self.crop_size:
                    image = image.resize(self.img_size, Image.BICUBIC)
                    label = label.resize(self.img_size, Image.NEAREST)
                    image = np.asarray(image, np.float32)
                    label = self.encode_segmap(np.array(label, dtype=np.uint8))
                else:
                    image = np.asarray(image, np.float32)
                    label = self.encode_segmap(np.array(label, dtype=np.uint8))
                    image = np.expand_dims(image, axis=2)
                    label = np.expand_dims(label, axis=2) 
                    img_label = np.concatenate((image, label), axis=-1)
                    img_label = CenterCrop(img_label,self.crop_size)
                    image = img_label[..., 0]
                    label = img_label[..., 1]
                    label = np.squeeze(label)
                    image = np.squeeze(image)

            elif self.resize_and_crop == "resize_and_center_crop":
                image = image.resize(self.img_size, Image.BICUBIC)
                image = np.asarray(image, np.float32)
                label = label.resize(self.img_size, Image.NEAREST)
                label = self.encode_segmap(np.array(label, dtype=np.uint8))
                label = np.expand_dims(label, axis=2) 
                img_label = np.concatenate((image, label), axis=-1)
                image, label = CenterCrop(image, label,self.crop_size)
                image = img_label[..., 0]
                label = img_label[..., 1]
                label = np.squeeze(label)
                image = np.squeeze(image)

            elif self.resize_and_crop == "resize_and_random_crop":
                image = image.resize(self.img_size, Image.BICUBIC)
                image = np.asarray(image, np.float32)

                label = label.resize(self.img_size, Image.NEAREST)
                label = self.encode_segmap(np.array(label, dtype=np.uint8))
                label = np.expand_dims(label, axis=2) 
                image = np.expand_dims(image, axis=2)
                img_label = np.concatenate((image, label), axis=-1)
                img_label = RandomCrop(img_label,self.crop_size)
                image = img_label[..., 0]
                label = img_label[..., 1]
                label = np.squeeze(label)
                image = np.squeeze(image)
        else:
            image = np.asarray(image, np.float32)
            label = self.encode_segmap(np.array(label, dtype=np.uint8))

        # normalize image
        image = image / 255.0
        image -= np.mean(image)
        image = image/np.std(image)

        label = np.expand_dims(label, axis=0) 
        image = np.expand_dims(image, axis=0)
        if self.nlabel:
            size = label.shape
            lbl = np.zeros((self.n_class,label.shape[1],label.shape[2]))
            for i in range(self.n_class):
                lbl[i] = np.array((label[0] == i), dtype=np.uint8)
        return image.copy(), lbl.copy(), label.copy(), size, name

    def encode_segmap(self, mask):
        for _validc in self.valid_classes:
            mask[mask==_validc] = self.class_map[_validc]
        return mask

class resizeSpinalcordDataSet(data.Dataset):
    def __init__(self, root,list_path, max_iters=None, img_size=128, crop_size=128,resize_and_crop='resize', batchsize=1, n_class=2,nlabel=True,set='train'):
        self.root = root
        self.list_path = list_path
        self.set = set
        self.batchsize = batchsize
        self.resize_and_crop = resize_and_crop
        self.n_class = n_class
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.img_size = (img_size,img_size)
        self.crop_size = crop_size
        self.valid_classes = [0, 128, 255]
        self.train_classes = [0, 1, 1]
        self.class_map = dict(zip(self.valid_classes, self.train_classes))
        self.files = []
        self.nlabel = nlabel
        self.img_masks = [name.strip().split(' ') for name in open(list_path)]
        for img_mask in self.img_masks:
            # use train
            img_file = osp.join(self.root, "%s/images/%s" % ('train', img_mask[0]))
            lbl_file = osp.join(self.root, "%s/mask/%s"%('train', img_mask[1]))
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
        label = Image.open(datafiles["lbl"])
        size = np.array(label).shape
        image,label = self.transform_imglabel(image,label,name)
        # normalize image
        image = image / 255.0
        image -= np.mean(image)
        image = image/np.std(image)
        label = np.expand_dims(label, axis=0) 
        image = np.expand_dims(image, axis=0)

        if self.nlabel:
            lbl = np.zeros((self.n_class,label.shape[1],label.shape[2]))
            for i in range(self.n_class):
                lbl[i] = np.array((label[0] == i), dtype=np.uint8)
        return image.copy(), lbl.copy(), label.copy(), size, name

    def transform_imglabel(self,image,label,name):
        size = np.array(image, dtype=np.uint8).shape
        # print(name)
        name1 = name[0]
        h,w = 2*size[0],2*size[1]
        if name1.startswith('site1') or name1.startswith('site2'):
            image = image.resize((w,h), Image.BICUBIC)
            label = label.resize((w,h), Image.NEAREST)
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

        image = np.expand_dims(image, axis=2)
        label = np.expand_dims(label, axis=2) 
        img_label = np.concatenate((image, label), axis=-1)
        img_label = CenterCrop(img_label,128)
        image = img_label[..., 0]
        label = img_label[..., 1]
        label = np.squeeze(label)
        image = np.squeeze(image)
        return image,label

    def encode_segmap(self, mask):
        for _validc in self.valid_classes:
            mask[mask==_validc] = self.class_map[_validc]
        return mask



class testResizeSpinalcordDataSet(data.Dataset):
    def __init__(self, root,list_path, max_iters=None, img_size=128, crop_size=128,resize_and_crop='resize', batchsize=1, n_class=2,nlabel=True,set='train'):
        self.root = root
        self.list_path = list_path
        self.set = set
        self.batchsize = batchsize
        self.resize_and_crop = resize_and_crop
        self.n_class = n_class
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.img_size = (img_size,img_size)
        self.crop_size = crop_size
        self.valid_classes = [0, 128, 255]
        self.train_classes = [0, 1, 1]
        self.class_map = dict(zip(self.valid_classes, self.train_classes))
        self.files = []
        self.nlabel = nlabel
        self.img_masks = [name.strip().split(' ') for name in open(list_path)]
        for img_mask in self.img_masks:
            # use train
            img_file = osp.join(self.root, "%s/images/%s" % ('test', img_mask[0]))

            lbl_file = osp.join(self.root, "%s/mask/%s"%('train', img_mask[1]))
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
        size = np.array(label).shape
        image, _ = self.transform_imglabel(image,label,name)
        label = self.encode_segmap(np.array(label, dtype=np.uint8))
        # normalize image
        image = image / 255.0
        image -= np.mean(image)
        image = image/np.std(image)
        label = np.expand_dims(label, axis=0) 
        image = np.expand_dims(image, axis=0)

        if self.nlabel:
            lbl = np.zeros((self.n_class,label.shape[1],label.shape[2]))
            for i in range(self.n_class):
                lbl[i] = np.array((label[0] == i), dtype=np.uint8)
        return image.copy(), lbl.copy(), label.copy(), size, name

    def transform_imglabel(self,image,label,name):
        size = np.array(image, dtype=np.uint8).shape
        # print(name)
        name1 = name[0]
        h,w = 2*size[0],2*size[1]
        if name1.startswith('site1') or name1.startswith('site2'):
            image = image.resize((w,h), Image.BICUBIC)
            label = label.resize((w,h), Image.NEAREST)
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

        image = np.expand_dims(image, axis=2)
        label = np.expand_dims(label, axis=2) 
        img_label = np.concatenate((image, label), axis=-1)
        img_label = CenterCrop(img_label,128)
        image = img_label[..., 0]
        label = img_label[..., 1]
        label = np.squeeze(label)
        image = np.squeeze(image)
        return image,label

    def encode_segmap(self, mask):
        for _validc in self.valid_classes:
            mask[mask==_validc] = self.class_map[_validc]
        return mask



# test nonlabel test data
class infertestResizeSpinalcordDataSet(data.Dataset):
    def __init__(self, root,list_path, max_iters=None, img_size=128, crop_size=128,resize_and_crop='resize', batchsize=1, n_class=2,nlabel=True,set='train'):
        self.root = root
        self.list_path = list_path
        self.set = set
        self.batchsize = batchsize
        self.resize_and_crop = resize_and_crop
        self.n_class = n_class
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.img_size = (img_size,img_size)
        self.crop_size = crop_size
        self.valid_classes = [0, 128, 255]
        self.train_classes = [0, 1, 1]
        self.class_map = dict(zip(self.valid_classes, self.train_classes))
        self.files = []
        self.nlabel = nlabel
        self.img_masks = [name.strip() for name in open(list_path)]
        for img_mask in self.img_masks:
            # use train
            img_file = osp.join(self.root, "%s/image/%s" % ('test', img_mask))
            lbl_file = img_file
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
        size = np.array(label).shape
        image, _ = self.transform_imglabel(image,label,name)
        label = self.encode_segmap(np.array(label, dtype=np.uint8))
        # normalize image
        image = image / 255.0
        image -= np.mean(image)
        image = image/np.std(image)
        label = np.expand_dims(label, axis=0) 
        image = np.expand_dims(image, axis=0)

        if self.nlabel:
            lbl = np.zeros((self.n_class,label.shape[1],label.shape[2]))
            for i in range(self.n_class):
                lbl[i] = np.array((label[0] == i), dtype=np.uint8)
        return image.copy(), lbl.copy(), label.copy(), size, name

    def transform_imglabel(self,image,label,name):
        size = np.array(image, dtype=np.uint8).shape
        # print(name)
        name1 = name
        if name1.startswith('site1'):
            image = image.resize((math.ceil(2*size[1]),math.ceil(2*size[0])), Image.BICUBIC)
            label = label.resize((math.ceil(2*size[1]),math.ceil(2*size[0])), Image.NEAREST)
            image = np.asarray(image, np.float32)
            print(image.shape)
            label = self.encode_segmap(np.array(label, dtype=np.uint8))
        elif name1.startswith('site2'):
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

        image = np.expand_dims(image, axis=2)
        label = np.expand_dims(label, axis=2) 
        img_label = np.concatenate((image, label), axis=-1)
        img_label = CenterCrop(img_label,128)
        image = img_label[..., 0]
        label = img_label[..., 1]
        label = np.squeeze(label)
        image = np.squeeze(image)
        return image,label

    def encode_segmap(self, mask):
        for _validc in self.valid_classes:
            mask[mask==_validc] = self.class_map[_validc]
        return mask

