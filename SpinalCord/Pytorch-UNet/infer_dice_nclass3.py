import torch
import os
from PIL import Image
import numpy as np


def calculate_dice(pred,mask):  
    pred = np.array(pred,dtype=np.uint8)
    mask = np.array(mask,dtype=np.uint8)
    pred = pred.reshape(-1,1)
    mask = mask.reshape(-1,1)
    eps = 0.0001
    inter = np.sum(pred*mask)
    union = np.sum(pred) + np.sum(mask) + eps
    dice = (2 * inter + eps) / union
    return dice

# def img_resize(mask):
#     mask_img = Image.fromarray(mask)
#     mask = mask_img.resize((48,64))
#     mask = np.array(mask)
#     return mask

def encode_segmap(mask):
    if np.sum(mask==127)>0:
        valid_classes = [0, 127, 255]
    else:
        valid_classes = [0, 128, 255]
    train_classes = [0, 1, 2]
    class_map = dict(zip(valid_classes, train_classes))
    for validc in valid_classes:
        mask[mask==validc] = class_map[validc]
    return mask

def get_padshape(image):
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

    print(hight,weight)
    pad_weight1 = int((weight - image.shape[1])/2)
    pad_weight2 = weight - image.shape[1] - pad_weight1
    pad_hight1 = int((hight - image.shape[0])/2)
    pad_hight2 = hight - image.shape[0] - pad_hight1
    return pad_weight1, pad_weight2, pad_hight1, pad_hight2


dice = np.array([0.0,0.0,0.0],dtype=np.float32)
s = np.array([0.0,0.0,0.0],dtype=np.float32)
target_path = "/media/jjchu/DataSets/spinalcord/train/cropmask/"
# pred_path = "/media/jjchu/seg/spinalcord/Results/UNet3/snapshots_site24_gen64_48_CP200_site13_save_orign_size/"
pred_path = "/media/jjchu/seg/spinalcord/Results/UNet3/snapshots_site24_gtcrop64_48_CP200_site13_save_orign_size/"
namelist = os.listdir(pred_path)
N_train = len(namelist)

for pred_name in namelist:
    # mask_name = '_'.join(pred_name.split('_')[:-1])+'.pgm'
    mask_name = pred_name
    pred = np.array(Image.open(os.path.join(pred_path, pred_name)).convert('L'),dtype=np.uint8)
    mask = np.array(Image.open(os.path.join(target_path, mask_name)).convert('L'),dtype=np.uint8)
    pred = encode_segmap(pred)
    mask = encode_segmap(mask)
    print(mask.shape)
    print(pred.shape)
    pad_weight1, pad_weight2, pad_hight1, pad_hight2 = get_padshape(mask)
    mask = np.pad(mask,((pad_hight1, pad_hight2),(pad_weight1, pad_weight2)),"constant")
    # mask = img_resize(mask)
    print(mask.shape)
    assert  pred.shape == mask.shape

    pred_lbl = np.zeros((3,pred.shape[0],pred.shape[1]))
    true_masks = np.zeros((3,mask.shape[0],mask.shape[1]))
    for i in range(3):
        pred_lbl[i] = np.array((pred == i), dtype=np.uint8)
        true_masks[i] = np.array((mask == i), dtype=np.uint8)

    for i in range(3):
        dice[i] = calculate_dice(pred_lbl[i],true_masks[i])

    s += dice
    print(dice)

dice_result = s/N_train
print(dice_result)
