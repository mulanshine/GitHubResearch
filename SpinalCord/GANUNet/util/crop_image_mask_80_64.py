import numpy as np 
import os
from PIL import Image
import cv2
import scipy.misc

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
    if size[0] <= 80 and size[1] <= 64:
        hight = 80
        weight = 64
        pad_weight1 = int((weight - size[1])/2)
        pad_weight2 = weight - size[1] - pad_weight1
        pad_hight1 = int((hight - size[0])/2)
        pad_hight2 = hight - size[0] - pad_hight1
    elif size[0] > 80 and size[1] <= 64:
        print("#######################>80or<48#######################################")
        print(size[0],size[1])
        weight = 64
        pad_weight1 = int((weight - size[1])/2)
        pad_weight2 = weight - size[1] - pad_weight1
        pad_hight1 = 0
        pad_hight2 = 0
    elif size[0] < 80 and size[1] > 64:
        print("#######################<80or>48#######################################")
        print(size[0],size[1])
        hight = 80
        pad_weight1 = 0
        pad_weight2 = 0
        pad_hight1 = int((hight - size[0])/2)
        pad_hight2 = hight - size[0] - pad_hight1
    elif size[0] > 80 and size[1] > 64:
        print("#######################>80or>48#######################################")
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
    return box,size



def crop_spinal_area_rect(imgpath,lblpath,filepath,imagesavepath,masksavepath):
    imgList = os.listdir(imgpath)
    lblList = os.listdir(lblpath)
    # 'site3-sc05-image_23.pgm site3-sc05-mask-r3_23.pgm\n'
    with open(filepath,"r") as f:
        nameList = f.readlines()

    for name in nameList:
        name = name.strip('\n')
        imgname = name[0]
        lblname = name[1]
        # imgname, lblname = name.split(' ')
        img = Image.open(os.path.join(imgpath,imgname))
       	mask = Image.open(os.path.join(lblpath,lblname))
        size = np.array(image, dtype=np.uint32).shape
        if imgname.startswith('site1') or imgname.startswith('site2'):
            image = image.resize((math.ceil(2.0*size[1]),math.ceil(2.0*size[0])), Image.BICUBIC)
            mask = mask.resize((math.ceil(2.0*size[1]),math.ceil(2.0*size[0])), Image.NEAREST)
            image = np.asarray(image, np.float32)
            mask = np.asarray(mask,np.uint8)
        elif imgname.startswith('site4'):
            image = image.resize((math.ceil(1.16*size[1]),math.ceil(1.16*size[0])), Image.BICUBIC)
            mask = mask.resize((math.ceil(1.16*size[1]),math.ceil(1.16*size[0])), Image.NEAREST)
            image = np.asarray(image, np.float32)
            mask = np.asarray(mask,np.uint8)
        elif imgname.startswith('site3'):
            image = np.array(image, dtype=np.float32)
            mask = np.array(mask, dtype=np.uint8)

        lbl = np.array(np.array(Image.open(os.path.join(lblpath,lblname)))>0,dtype=np.uint8)
        box,size = get_area_rect(lbl)
        box,size = get_area_fixed_size(box,size)
        # imglbl = img*lbl
        imglbl = img
        img_top, img_left = box[1][0]-1, box[1][1]-1
        th, tw = size[0],size[1]
        imagecontainer = np.zeros((size[0], size[1], imglbl.shape[-1]), np.float32)
        imagecontainer = imglbl[img_top:img_top+th, img_left:img_left+tw]
        maskcontainer = np.zeros((size[0], size[1], imglbl.shape[-1]), np.float32)
        maskcontainer = mask[img_top:img_top+th, img_left:img_left+tw]

        scipy.misc.imsave(os.path.join(imagesavepath,lblname),imagecontainer)
        scipy.misc.imsave(os.path.join(masksavepath,lblname), maskcontainer)
        # print(name)


imgpath = "/home/jjchu/spinalcord/images/"
lblpath = "/home/jjchu/spinalcord/mask/"
filepath = "/home/jjchu/MyResearch/SpinalCord/Pytorch-UNet/data/train_site12_list.txt"
imagesavepath = "/home/jjchu/spinalcord/test/image_crop/"
masksavepath = "/home/jjchu/spinalcord/test/mask_crop/"
crop_spinal_area_rect(imgpath,lblpath,filepath,imagesavepath,masksavepath)

