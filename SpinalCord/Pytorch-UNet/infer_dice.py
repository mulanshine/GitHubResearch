import torch
import os
from PIL import Image
import numpy as np
import numbers

def encode_segmap(mask):
    valid_classes = [0, 128, 255]
    train_classes = [0, 1, 1]
    class_map = dict(zip(valid_classes, train_classes))
    for _validc in valid_classes:
        mask[mask==_validc] = class_map[_validc]
    return mask


def centerCrop(imgarr, size):
    h, w = imgarr.shape[0],imgarr.shape[1]
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        size = size
    th, tw = size
    img_left = int(round((w - tw) / 2.))
    img_top = int(round((h - th) / 2.))
    container = np.zeros((th, tw), np.float32)
    container= imgarr[img_top:img_top+th, img_left:img_left+tw]
    return container

def centerCrop1(imgarr, size):
    h, w = imgarr.shape[0],imgarr.shape[0]
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        size = size
    th, tw = size
    img_left = int(round((w - tw) / 2.))
    img_top = int(round((h - th) / 2.))
    container = np.zeros((th, tw), np.float32)
    container= imgarr[img_left:img_left+tw,img_top:img_top+th]
    return container


def transform_mask(label):
    size = np.array(label, dtype=np.uint8).shape[1]
    if size > 128:
        # label = encode_segmap(np.array(label, dtype=np.uint8))
    # else:
        # label = encode_segmap(np.array(label, dtype=np.uint8))
        label = centerCrop(label,128)
    return label
    
dice = 0.0
S1,S2,S3,S4 = 0.0,0.0,0.0,0.0
N1,N2,N3,N4 = 0,0,0,0
target_path = "/media/jjchu/DataSets/spinalcord/train/mask/"
# pred_path = '/media/jjchu/seg/spinalcord/Results/UNet2/unet_test_site13_centercrop128_CP90/'
pred_path = "/media/jjchu/seg/spinalcord/Results/UNet2/unet_test_site13_centercrop128_lr_grey_CP90/"

namelist = os.listdir(pred_path)
N_train = len(namelist)

for pred_name in namelist:
    # mask_name = '_'.join(pred_name.split('_')[:-1])+'.pgm'
    mask_name = pred_name
    pred = np.array(np.array(Image.open(os.path.join(pred_path, pred_name)).convert('L'),dtype=np.uint8)>0,dtype=np.uint8)
    mask = np.array(np.array(Image.open(os.path.join(target_path, mask_name)).convert('L'),dtype=np.uint8)>0,dtype=np.uint8)
    mask = transform_mask(mask)
    assert  pred.shape == mask.shape
    # print(pred.shape,mask.shape)
    pred = pred.reshape(-1,1)
    mask = mask.reshape(-1,1)
    eps = 0.0001
    inter = np.sum(pred*mask)
    union = np.sum(pred) + np.sum(mask) + eps
    dice = (2 * inter + eps) / union
    if pred_name.startswith('site1'):
        N1 += 1
        S1 += dice
    elif pred_name.startswith('site2'):
        N2 += 1
        S2 += dice
    if pred_name.startswith('site3'):
        N3 += 1
        S3 += dice
        print(dice)
    if pred_name.startswith('site4'):
        N4 += 1
        S4 += dice


dice1 = S1/N1
dice2 = S2/N2
dice3 = S3/N3
dice4 = S4/N4
print(N1,dice1,N2,dice2,N3,dice3,N4,dice4)
print(pred_path)


