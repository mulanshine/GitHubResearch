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


# train_pairs_list = "/home/jjchu/My_Research/spinal_cord_segmentation/data/train_site12_list.txt"
# site2-sc04-image_1.pgm site2-sc04-mask-r3_1.pgm
# root = "/public/jjchu/Dataset/Spinel_cord_segmentation/"train/|mask|images|
class spinalcordDataSet(data.Dataset):
    def __init__(self, root,list_path, max_iters=None, img_size=360, crop_size=256,resize_and_crop='resize', batchsize=1, n_class=2,nlabel=True,set='train'):
    # def __init__(self, root,list_path, max_iters=None, n_class=2, set='train'):

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
        self.train_classes = [0, 1, 2]
        self.class_map = dict(zip(self.valid_classes, self.train_classes))
        self.files = []
        self.nlabel = nlabel
        self.img_masks = [name.strip().split(' ') for name in open(list_path)]
        for img_mask in self.img_masks:
            img_file = osp.join(self.root, "%s/images/%s" % (self.set, img_mask[0]))
            if self.set == "train":
                lbl_file = osp.join(self.root, "%s/mask/%s"%(self.set, img_mask[1]))
            else:
                lbl_file = osp.join(self.root, "%s/mask/%s"%(self.set, img_mask[0]))
            self.files.append({
                "img": img_file,
                "lbl": lbl_file,
                "name": img_mask
            })        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        if self.set =='train':   
            if label.shape[0] > 32 and label.shape[1] > 24:
                height = 64
                weight = 48
            elif label.shape[0] < 32 and label.shape[1] < 24:
                height = 32
                weight = 24
            else:
                height = 64
                weight = 48

        image = Image.open(datafiles["img"]).convert('L')
        image = np.asarray(image, np.float32)
        image = np.expand_dims(image, axis=2) 
        name = datafiles["name"]
        label = Image.open(datafiles["lbl"])
        label = self.encode_segmap(np.array(label, dtype=np.uint8))
        label = np.expand_dims(label, axis=2) 
        img_label = np.concatenate((image, label), axis=-1)

        pad_weight1 = int((weight - label.shape[0])/2)
        pad_weight2 = weight - label.shape[0] - pad_weight1
        pad_hight1 = int((hight - label.shape[1])/2)
        pad_hight2 = hight - label.shape[1] - pad_hight1
        img_label = np.pad(img_label,(pad_hight1, pad_hight2),(pad_weight1, pad_weight2),(0,0)),"constant"）
   
        # img_label = RandomCrop(img_label,self.crop_size)
        image = img_label[..., :3]
        image = np.squeeze(image)
        label = img_label[..., 3:]
        label = np.squeeze(label)

        size = image.shape[:2]
        image = image[:, :, ::-1]  # change to BGR
        # normalize image
        image = image / 255.0
        image -= np.array([0.485, 0.456, 0.406])
        image = image / np.array([0.229, 0.224, 0.225])
        # image = image.transpose((2, 0, 1)).astype(np.float32)
        relation_map = ExtractAffinityLabelInRadius(image)
        
        if self.nlabel:
            size = label.shape
            lbl = np.zeros((self.n_class,label.shape[0],label.shape[1]))
            for i in range(self.n_class):
                lbl[i] = np.array((label == i), dtype=np.uint8)
            label = lbl

        return image.copy(), label.copy(),relation_map, size, name

    def encode_segmap(self, mask):
        for _validc in self.valid_classes:
            mask[mask==_validc] = self.class_map[_validc]
        return mask


    def ExtractAffinityLabelInRadius(image):
        images_from = image
        images_pad = np.pad(image,((1,1),(1,1)),'constant')
        images_to = np.zeros((8,image.shape[0],imsge.shape[1]))
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

        relation_map = diff_maps.copy()
        return relation_map
