import numpy as np 
import os
from PIL import Image
import cv2
import scipy.misc
import math


# get the minimum enclosing rectangle (center(x,y),(weight,hight),rotated angle)
# get the four points
def get_min_area_rect(impath):
    img = np.array(np.array(Image.open(impath))>0,dtype=np.uint8)
    itemindex = np.argwhere(img == 1)
    rect = cv2.minAreaRect(itemindex) 
    box = cv2.boxPoints(rect) 
    return box
 
def get_area_rect(img):
    itemindex = np.argwhere(img == 1)
    X = itemindex[:,0]
    Y = itemindex[:,1]
    minX,maxX = np.min(X), np.max(X)
    minY,maxY = np.min(Y), np.max(Y)
    size = [maxX - minX + 1,maxY - minY + 1]
    box = [[minX,maxY],[minX,minY],[maxX,minY],[maxX,maxY]]
    return box,size

def get_area_fixed_size(box,size):
    eh,ew = 128,128
    if size[0] <= eh and size[1] <= ew:
        hight = eh
        weight = ew
        pad_weight1 = int((weight - size[1])/2)
        pad_weight2 = weight - size[1] - pad_weight1
        pad_hight1 = int((hight - size[0])/2)
        pad_hight2 = hight - size[0] - pad_hight1
    elif size[0] > eh and size[1] <= ew:
        print("#######################>80or<64#######################################")
        print(size[0],size[1])
        weight = ew
        pad_weight1 = int((weight - size[1])/2)
        pad_weight2 = weight - size[1] - pad_weight1
        pad_hight1 = 0
        pad_hight2 = 0
    elif size[0] < eh and size[1] > ew:
        print("#######################<80or>64#######################################")
        print(size[0],size[1])
        hight = eh
        pad_weight1 = 0
        pad_weight2 = 0
        pad_hight1 = int((hight - size[0])/2)
        pad_hight2 = hight - size[0] - pad_hight1
    elif size[0] > eh and size[1] > ew:
        print("#######################>80or>64#######################################")
        print(size[0],size[1])
        pad_weight1 = 0
        pad_weight2 = 0
        pad_hight1 = 0
        pad_hight2 = 0
    [[minX,maxY],[minX,minY],[maxX,minY],[maxX,maxY]] = box
    minX = minX - pad_hight1
    maxX = maxX + pad_hight2
    minY = minY - pad_weight1
    maxY = maxY + pad_weight2
    size = [maxX - minX + 1,maxY - minY + 1]
    box = [[minX,maxY],[minX,minY],[maxX,minY],[maxX,maxY]]
    # print(minX,maxX,minY,maxY)
    return box,size

def encodemap(mask,thr):
    mask = np.array(mask,dtype=np.float32)
    # new_mask = np.zeros(mask.shape)
    mask255 = abs(255 - mask) # + 10
    mask128 = abs(mask - 128) + 40
    mask0 = abs(mask - 0) #+ 20
    mask255 = np.expand_dims(mask255,0)
    mask128 = np.expand_dims(mask128,0)
    mask0 = np.expand_dims(mask0,0)
    mask3 = np.concatenate((mask0,mask128,mask255),0)#
    mask3 = np.argmin(mask3,axis=0)
    mask3 = np.array(mask3,dtype=np.uint8)

    valid_classes = [0, 128, 255] #
    train_classes = [0, 1, 2] #
    class_map = dict(zip(train_classes, valid_classes))
    for validc in train_classes:
        mask[mask3==validc] = class_map[validc]

    for i in range(mask.shape[0]):
        for j in range(1,mask.shape[1]):
            if mask[i,j]==128:
                if mask[i,j-1]==0  or mask[i,j+1]==0:
                    mask[i,j]=255
    
    return mask


# def encodemap(mask,thr):
#     mask = mask.copy()
#     mask[mask>=thr] = 128
#     mask[mask<thr] = 0
#     return mask

# for i in range(mask.shape[0]):
#     for j in range(mask.shape[1]):
#         new_mask = np.zeros(mask.shape)
#         if abs(mask[i,j] - 255) < abs(mask[i,j] - 128) and abs(mask[i,j] - 255) < abs(mask[i,j] - 0):
#             new_mask[i,j] = 255
#         elif abs(mask[i,j] - 128) < abs(mask[i,j] - 255) and abs(mask[i,j] - 128) < abs(mask[i,j] - 0):
#             new_mask[i,j] = 128
#         elif abs(mask[i,j] - 0) < abs(mask[i,j] - 128) and abs(mask[i,j] - 0) < abs(mask[i,j] - 255):
#             new_mask[i,j] = 0
# return mask

# for i in range(mask.shape[0]):
#     for j in range(mask.shape[1]):
#         if mask[i,j]<172 and mask[i,j]>64:
#             mask[i,j]=128
#         elif mask[i,j]>=172 and mask[i,j]<=255:
#             mask[i,j]=255
#         elif mask[i,j]<=64 and mask[i,j]>=0:
#             mask[i,j]=0

# return mask


def crop_spinal_area_rect(imgpath,lblpath,filepath,imagesavepath,masksavepath):
    imgList = os.listdir(imgpath)
    lblList = os.listdir(lblpath)
    # 'site3-sc05-image_23.pgm site3-sc05-mask-r3_23.pgm\n'
    with open(filepath,"r") as f:
        nameList = f.readlines()

    for name in nameList:
        name = name.strip('\n')
        # imgname = name[0]
        # lblname = name[1]
        imgname, lblname = name.split(' ')
        image = Image.open(os.path.join(imgpath,imgname))
       	mask = Image.open(os.path.join(lblpath,lblname))
        size = np.array(image, dtype=np.uint32).shape
        mask = np.array(mask,dtype=np.uint8)
        valid_classes = [0, 128, 255]
        train_classes = [0, 128, 255]
        class_map = dict(zip(valid_classes, train_classes))
        for validc in valid_classes:
            mask[mask==validc] = class_map[validc]
        # print(mask.shape)
        mask = Image.fromarray(mask)

        if imgname.startswith('site1') or imgname.startswith('site2'):
            image = image.resize((math.ceil(2.0*size[1]),math.ceil(2.0*size[0])), Image.BICUBIC)
            mask = mask.resize((math.ceil(2.0*size[1]),math.ceil(2.0*size[0])), Image.NEAREST) # Image.NEAREST,ANTIALIAS
            image = np.asarray(image, np.float32)
            mask = np.asarray(mask,np.uint8)
        elif imgname.startswith('site4'):
            image = image.resize((math.ceil(1.16*size[1]),math.ceil(1.16*size[0])), Image.BICUBIC)
            mask = mask.resize((math.ceil(1.16*size[1]),math.ceil(1.16*size[0])), Image.NEAREST)# Image.NEAREST,ANTIALIAS
            image = np.asarray(image, np.float32)
            mask = np.asarray(mask,np.uint8)
        elif imgname.startswith('site3'):
            image = np.array(image, dtype=np.float32)
            mask = np.array(mask, dtype=np.uint8)

        # if imgname.startswith('site1') or imgname.startswith('site2') or imgname.startswith('site4'):
        #     mask = encodemap(mask,100)
        # mask = np.array(mask, dtype=np.uint8)

        # im.resize(size, Image.ANTIALIAS)
        # print(mask.shape)
        lbl = np.array(np.array(mask)>0,dtype=np.uint8)
        box,size = get_area_rect(lbl)
        box,size = get_area_fixed_size(box,size)
        # imglbl = img*lbl
        imglbl = image
        img_top, img_left = box[1][0]-1, box[1][1]-1
        th, tw = size[0],size[1]
        print(th,tw)
        imagecontainer = np.zeros((size[0], size[1], imglbl.shape[-1]), np.float32)
        imagecontainer = imglbl[img_top:img_top+th, img_left:img_left+tw]
        maskcontainer = np.zeros((size[0], size[1], imglbl.shape[-1]), np.float32)
        maskcontainer = mask[img_top:img_top+th, img_left:img_left+tw]

        scipy.misc.imsave(os.path.join(imagesavepath,lblname),imagecontainer)
        scipy.misc.imsave(os.path.join(masksavepath,lblname), maskcontainer)
        # break
        # print(name)


imgpath = "/home/jjchu/DataSet/spinalcord/images/"
lblpath = "/home/jjchu/DataSet/spinalcord/mask/"
filepath = "/home/jjchu/GitHubResearch/SpinalCord/Pytorch-UNet/data/train_site12_list.txt"
imagesavepath = "/home/jjchu/DataSet/spinalcord/crop_128/image_crop/"
masksavepath = "/home/jjchu/DataSet/spinalcord/crop_128/mask_crop/"
crop_spinal_area_rect(imgpath,lblpath,filepath,imagesavepath,masksavepath)

