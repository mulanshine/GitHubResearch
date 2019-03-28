import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import math 


"""segcgan_80_64_meansite12_denoise.py"""

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
    # check
    assert data.size == data_gm.size, "Data to normalized and GM data do not have the same shape."
    assert data.size == data_wm.size, "Data to normalized and WM data do not have the same shape."
    # avoid shape error because of 3D-like shape for 2D (x, x, 1) instead of (x,x)
    data_gm = data_gm.reshape(data.shape)
    data_wm = data_wm.reshape(data.shape)
    # put almost zero background to zero
    data[data < 1e-10] = 0
    # binarize GM and WM data if needed
    if np.min(data_gm) != 0 or np.max(data_gm) != 1:
        data_gm[data_gm < 0.5] = 0
        data_gm[data_gm >= 0.5] = 1
    if np.min(data_wm) != 0 or np.max(data_wm) != 1:
        data_wm[data_wm < 0.5] = 0
        data_wm[data_wm >= 0.5] = 1
    # get GM and WM values in slice
    data_in_gm = data[data_gm == 1]
    data_in_wm = data[data_wm == 1]
    med_data_gm = np.median(data_in_gm)
    med_data_wm = np.median(data_in_wm)

    assert med_data_gm != med_data_wm, "med_data_gm = med_data_wm."

    std_data = np.mean([np.std(data_in_gm), np.std(data_in_wm)])
    # compute normalized data
    # if median values are too close: use min and max to normalize data
    if abs(med_data_gm - med_data_wm) < std_data and val_min is not None and val_max is not None:
        try:
            min_data = min(np.min(data_in_gm[data_in_gm.nonzero()]), np.min(data_in_wm[data_in_wm.nonzero()]))
            max_data = max(np.max(data_in_gm[data_in_gm.nonzero()]), np.max(data_in_wm[data_in_wm.nonzero()]))
            new_data = ((data - min_data) * (val_max - val_min) / (max_data - min_data)) + val_min
        except ValueError:
            printv('WARNING: an incomplete slice will not be normalized', 1, 'warning')
            return data
    # else (=normal data): use median values to normalize data
    else:
        new_data = ((data - med_data_wm) * (val_gm - val_wm) / (med_data_gm - med_data_wm)) + val_wm
    # put almost zero background to zero
    new_data[data < 1e-10] = 0  # put at 0 the background
    # new_data[new_data < 1e-10] = 0  # put at 0 the background
    # return normalized data
    return new_data

def mean_std_norm_slice(data, lbls, val_gm, val_wm, std_gm, std_wm):
    data_gm, data_wm,data_bk = lbls[1],lbls[2],lbls[0]
    data[data < 1e-10] = 0
    # get GM and WM values in slice
    data_in_gm = data[data_gm == 1]
    data_in_wm = data[data_wm == 1]
    med_data_gm = np.mean(data_in_gm)
    med_data_wm = np.mean(data_in_wm)
    std_data_gm = np.std(data_in_gm)
    std_data_wm = np.std(data_in_wm)
    assert med_data_gm != med_data_wm, "med_data_gm = med_data_wm."
    data_gm_im = data * data_gm
    data_wmbk_im = data * (data_wm+data_bk)
    # data_bk_im = data * data_bk
    new_data_gm_im = (data_gm_im - med_data_gm)/std_data_gm*std_gm + val_gm 
    new_data_wmbk_im = (data_wmbk_im - med_data_wm)/std_data_wm*std_wm + val_wm 
    new_data = new_data_gm_im * data_gm + new_data_wm_im * data_wm 
    # new_data = ((data - med_data_wm) * (val_gm - val_wm) / (med_data_gm - med_data_wm)) + val_wm
    # put almost zero background to zero
    new_data[data < 1e-10] = 0  # put at 0 the background
    # new_data[new_data < 1e-8] = 0  # put at 0 the background
    # return normalized data
    return new_data


# use(80,48),resize to (80,48) for generator
class SegcganDataset(BaseDataset):
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
        self.valid_classes = [0, 128, 255]
        self.train_classes = [0, 1, 2]
        self.class_map = dict(zip(self.valid_classes, self.train_classes))
        self.val_gm = 0.51590 # 131.5549/255.0=0.515901568627451
        self.val_wm = 0.45518 # 116.0709/255.0
        self.std_gm = 0.03399
        self.std_wm = 0.04475

    def __getitem__(self, index):
        imgpath = self.imgpaths[index]
        name = imgpath.split('/')[-1]
        maskpath = os.path.join(self.maskdir,name)
        mask = Image.open(maskpath).convert('L')
        imgL = Image.open(imgpath).convert('L')
        imgL = self.imgtransform_L(imgL,name)
        mask = self.imgtransform_mask(mask,name)
        #
        # mask = torch.FloatTensor(mask)
        # mask = mask.long()
        # imgRGB = Image.open(imgpath).convert('RGB')
        # imgRGB = (np.transpose(np.array(imgRGB,np.float32),(2,0,1))/255.0 - 0.5) * 2
        # imgRGB = torch.FloatTensor(imgRGB)
        #
        imgRGB, mask = self.transtype(imgL, mask)
        # aff = self.binary_relation_map(imgL)
        aff = self.compute_relation_map(imgL)
        return {'img': imgRGB, 'lbl':mask, 'aff':aff, 'path': imgpath}

    def transtype(self,imgL, mask):
        lbls = self.crop_GM_WM(mask)
        img = normalize_slice(imgL[0], lbls, self.val_gm, self.val_wm)
        # img = mean_std_norm_slice(imgL[0], lbls, self.val_gm, self.val_wm, self.std_gm, self.std_wm)
        img = np.expand_dims(img, axis=0)
        imgRGB = np.concatenate((img,img,img),axis=0)
        imgRGB = (imgRGB - 0.5) * 2
        mask = torch.FloatTensor(mask)
        mask = mask.long()
        imgRGB = torch.FloatTensor(imgRGB)
        return imgRGB, mask

    def encode_segmap(self, mask):
        for validc in self.valid_classes:
            mask[mask==validc] = self.class_map[validc]
        return mask

    def crop_GM_WM(self, label,n_class=3):
        label = self.encode_segmap(np.array(label, dtype=np.uint8))
        # lbls[0]:back,lbls[1]:grey,lbls[2]:white
        lbls = np.zeros((n_class,label.shape[-2],label.shape[-1]))
        for i in range(n_class): # 0,1,2
            lbls[i] = np.array((label == i), dtype=np.uint8)
        return lbls

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

    def imgtransform_L(self, image, name,denoising=1):
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

        image = image / 255.0
        pad_weight1, pad_weight2, pad_hight1, pad_hight2 = self.get_padshape(image)
        image = np.pad(image,((pad_hight1, pad_hight2),(pad_weight1, pad_weight2)),"constant")
        img = np.expand_dims(image, axis=0)
        imgRGB = np.concatenate((img,img,img),axis=0)
        if denoising == 1:
            imgRGB = np.transpose(imgRGB,(1,2,0))
            imgRGB = denoise_nlmeans(imgRGB)
            imgRGB = np.transpose(imgRGB,(2,0,1))
            image = imgRGB[0]
            image = np.expand_dims(image, axis=0)
        else:
            image = np.expand_dims(image, axis=0)
        return image


    def imgtransform_mask(self, label, name):
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

        pad_weight1, pad_weight2, pad_hight1, pad_hight2 = self.get_padshape(label)
        label = np.pad(label,((pad_hight1, pad_hight2),(pad_weight1, pad_weight2)),"constant")
        return label

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

    def compute_relation_map(self, image,denoising=1):
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
        return len(self.imgpaths)

    def name(self):
        return 'SegcganDataset'


