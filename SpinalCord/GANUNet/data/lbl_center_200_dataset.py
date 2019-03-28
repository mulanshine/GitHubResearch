import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import math 
import random
import numbers

"""lbl_center_200_dataset.py"""

def CenterCrop(imgarr, size):
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

def denoise_nlmeans(data_in, patch_radius=1, block_radius=5):
    """
    data_in: nd_array to denoise
    for more info about patch_radius and block radius, please refer to the dipy website: http://nipy.org/dipy/reference/dipy.denoise.html#dipy.denoise.nlmeans.nlmeans
    """
    from dipy.denoise.nlmeans import nlmeans
    from dipy.denoise.noise_estimate import estimate_sigma
    from numpy import asarray
    data_in = asarray(data_in)
    block_radius_max = min(data_in.shape) - 1
    block_radius = block_radius_max if block_radius > block_radius_max else block_radius
    sigma = estimate_sigma(data_in)
    denoised = nlmeans(data_in, sigma, patch_radius=patch_radius, block_radius=block_radius)
    return denoised

def normalize_slice(data, lbls, val_gm, val_wm, val_min=None, val_max=None):
    data_gm, data_wm = lbls[1],lbls[2]
    data = data[0]
    data_gwm = lbls[2]+lbls[1]
    data_bk = data * (lbls[0])
    # check
    # assert data.size == data_gm.size, "Data to normalized and GM data do not have the same shape."
    # assert data.size == data_wm.size, "Data to normalized and WM data do not have the same shape."
    # avoid shape error because of 3D-like shape for 2D (x, x, 1) instead of (x,x)
    # data_gm = data_gm.reshape(data.shape)
    # data_wm = data_wm.reshape(data.shape)
    # put almost zero background to zero
    # data[data < 1e-10] = 0

    # get GM and WM values in slice
    data_in_gm = data[data_gm == 1]
    data_in_wm = data[data_wm == 1]
    # data_in_bk = data[data_bk == 1]
    med_data_gm = np.median(data_in_gm)
    med_data_wm = np.median(data_in_wm)
    # med_data_bk = np.median(data_in_bk)
    if med_data_gm == med_data_wm:
        print("med_data_gm = med_data_wm")
        med_data_gm = np.mean(data_in_gm)
        med_data_wm = np.mean(data_in_wm)
        assert med_data_gm != med_data_wm, "med_data_gm = med_data_bk."

    new_data = ((data - med_data_wm) * (val_gm - val_wm) / (med_data_gm - med_data_wm + 1e-10)) + val_wm
    new_data[new_data > 1] = 1
    new_data = new_data * data_gwm + data_bk
    # put almost zero background to zero
    new_data[data < 1e-10] = 0  # put at 0 the background
    # new_data[new_data < 1e-10] = 0  # put at 0 the background
    # return normalized data
    # new_data = data_bk
    img = np.expand_dims(new_data, axis=0)
    new_image = np.concatenate((img,img,img),axis=0)
    return new_image


def normalize_slice2(data, mask, val_gm, val_wm, val_min=None, val_max=None):
    mask = Image.fromarray(mask)
    size = np.array(mask, dtype=np.uint8).shape
    mask2 = mask.resize((math.ceil(2.0*size[1]),math.ceil(2.0*size[0])), Image.NEAREST)
    mask2 = np.array(mask2,dtype=np.uint8)
    mask2 = CenterCrop(mask, 200)
    lbls = np.zeros((n_class,2*mask.shape[-2],2*mask.shape[-1]))
    lbls2 = np.zeros((n_class,2*mask2.shape[-2],2*mask2.shape[-1]))
    for i in range(n_class): # 0,1,2
        lbls[i] = np.array((mask == i), dtype=np.uint8)
        lbls2[i] = np.array((mask2 == i), dtype=np.uint8)
    
    data_gm, data_wm = lbls[1],lbls[2]
    data2_gw = data*(lbls2[1]+lbls2[2])
    data = data[0]
    data_gwm = lbls[2]+lbls[1]
    data_bk = data * (lbls[0])

    # get GM and WM values in slice
    data_in_gm = data[data_gm == 1]
    data_in_wm = data[data_wm == 1]
    # data_in_bk = data[data_bk == 1]
    med_data_gm = np.median(data_in_gm)
    med_data_wm = np.median(data_in_wm)
    # med_data_bk = np.median(data_in_bk)
    if med_data_gm == med_data_wm:
        print("med_data_gm = med_data_wm")
        med_data_gm = np.mean(data_in_gm)
        med_data_wm = np.mean(data_in_wm)
        assert med_data_gm != med_data_wm, "med_data_gm = med_data_bk."

    new_data = ((data - med_data_wm) * (val_gm - val_wm) / (med_data_gm - med_data_wm + 1e-10)) + val_wm
    new_data[new_data > 1] = 1
    new_data = new_data * data_gwm + data_bk
    # put almost zero background to zero
    new_data[data < 1e-10] = 0  # put at 0 the background
    # new_data[new_data < 1e-10] = 0  # put at 0 the background
    # return normalized data
    # new_data = data_bk
    img = np.expand_dims(new_data, axis=0)
    new_image = np.concatenate((img,img,img),axis=0)
    return new_image

def mean_std_norm_slice(data, lbls, val_gm, val_wm, std_gm, std_wm):
    data_gm, data_wm,data_bk = lbls[1],lbls[2],lbls[0]
    data[data < 1e-10] = 0
    # get GM and WM values in slice
    data_in_gm = data[data_gm == 1]
    data_in_wm = data[data_wm == 1]
    med_data_gm = np.median(data_in_gm)
    med_data_wm = np.median(data_in_wm)
    assert med_data_gm != med_data_wm, "med_data_gm = med_data_wm."
    data_gm_im = data * data_gm
    data_wmbk_im = data * (data_wm + data_bk)

    new_data_gm_im = (data_gm_im - med_data_gm) + val_gm 
    new_data_wmbk_im = (data_wmbk_im - med_data_wm) + val_wm 
    new_data = new_data_gm_im * data_gm + new_data_wmbk_im * (data_wm + data_bk)
    # new_data = ((data - med_data_wm) * (val_gm - val_wm) / (med_data_gm - med_data_wm)) + val_wm
    # put almost zero background to zero
    new_data[data < 1e-10] = 0  # put at 0 the background
    # new_data[new_data < 1e-8] = 0  # put at 0 the background
    # return normalized data
    return new_data


class LblCGANDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.phase = opt.phase
        self.imgdir = os.path.join(opt.dataroot,'images')
        self.maskdir = os.path.join(opt.dataroot,'mask')
        self.maskpaths = sorted(make_dataset(self.maskdir,site=self.opt.site))
        self.img_size = 200
        self.valid_classes = [0, 128, 255]
        if self.opt.n_class == 3:
            self.train_classes = [0, 1, 2]
        elif self.opt.n_class == 2:
            self.train_classes = [0, 1, 1]
        self.class_map = dict(zip(self.valid_classes, self.train_classes))
        self.val_gm = 0.51590 # 131.5549/255.0=0.515901568627451+0.05
        self.val_wm = 0.45518 # wm=116.0709/255.0
        self.val_bk = 0.30961 
        # self.std_gm = 0.03399
        # self.std_wm = 0.04475
        # mean 572 0.3096077718292848 0.5125359476090164
        # median 572 0.3059406377948247 0.5139757478414299

    def __getitem__(self, index):
        maskpath = self.maskpaths[index]
        maskname = maskpath.split('/')[-1]
        imgname = "-".join(maskname.replace("mask", "image").split('-')[:3]) + "_" + maskname.split('_')[-1]
        imgpath = os.path.join(self.imgdir,imgname)
        image = Image.open(imgpath).convert('RGB')
        label = Image.open(maskpath).convert('L')
        image,label = self.transform(image, label, maskname, denoising=self.opt.imgdenoising)
        real = (image-0.5)*2
        image, label = self.transtype(image, label)
        return {'real': image, 'label':label, 'path': maskpath}

    def transtype(self,image, mask):
        # for image
        image = (image-0.5)*2
        image = torch.FloatTensor(image)
        # for mask
        mask = np.expand_dims(mask,axis=0)
        mask = torch.FloatTensor(mask)
        return image, mask

    def encode_segmap(self, mask):
        for validc in self.valid_classes:
            mask[mask==validc] = self.class_map[validc]
        return mask

    def crop_GM_WM(self, label,n_class=3):   
        # lbls[0]:back,lbls[1]:grey,lbls[2]:white
        lbls = np.zeros((n_class,label.shape[-2],label.shape[-1]))
        for i in range(n_class): # 0,1,2
            lbls[i] = np.array((label == i), dtype=np.uint8)
        return lbls

    def transform(self, image, label,name,denoising=0):
        size = np.array(image, dtype=np.uint8).shape
        if name.startswith('site1') or name.startswith('site2'):
            image = image.resize((math.ceil(2.0*size[1]),math.ceil(2.0*size[0])), Image.BICUBIC)
            label = label.resize((math.ceil(2.0*size[1]),math.ceil(2.0*size[0])), Image.NEAREST)
            image = np.asarray(image, np.float32)
            label = self.encode_segmap(np.array(label, dtype=np.uint8))
        elif name.startswith('site4'):
            image = image.resize((math.ceil(1.16*size[1]),math.ceil(1.16*size[0])), Image.BICUBIC)
            label = label.resize((math.ceil(1.16*size[1]),math.ceil(1.16*size[0])), Image.NEAREST)
            image = np.asarray(image, np.float32)
            label = self.encode_segmap(np.array(label, dtype=np.uint8))
        elif name.startswith('site3'):
            image = np.array(image, dtype=np.float32)
            label = self.encode_segmap(np.array(label, dtype=np.uint8))

        label = np.expand_dims(label, axis=2) 
        img_label = np.concatenate((image, label), axis=-1)
        img_label = CenterCrop(img_label,self.img_size)
        image = img_label[..., 0:3]
        image = np.squeeze(image)
        label = img_label[..., 3]
        label = np.squeeze(label)
        # label = np.expand_dims(label, axis=0)
        image = image / 255.0
        if denoising == 1:
            image = denoise_nlmeans(image)
        image = np.transpose(image,(2,0,1))
        return image,label


    def compute_relation_map(self, image,denoising=0):
        # image = (image - 0.5) * 2
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
        if denoising == 1:
            relation_map = np.transpose(diff_maps,(1,2,0))
            relation_map = denoise_nlmeans(relation_map)
            relation_map = np.transpose(relation_map,(2,0,1))
        relation_map = (diff_maps - diff_maps.min()) / (diff_maps.max()-diff_maps.min()+1e-10) * 2.0
        relation_map = relation_map - 1.0
        relation_map = torch.FloatTensor(relation_map)
        return relation_map

    def __len__(self):
        return len(self.maskpaths)

    def name(self):
        return 'LblCGANDataset'
