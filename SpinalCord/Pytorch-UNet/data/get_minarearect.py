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


def crop_spinal_area_rect(imgpath,lblpath,filepath,imagesavepath,masksavepath):
    imgList = os.listdir(imgpath)
    lblList = os.listdir(lblpath)
    # 'site3-sc05-image_23.pgm site3-sc05-mask-r3_23.pgm\n'
    with open(filepath,"r") as f:
        nameList = f.readlines()

    for name in nameList:
        name = name.strip('\n')
        imgname = name
        lblname = name
        # imgname, lblname = name.split(' ')
        img = np.array(Image.open(os.path.join(imgpath,imgname)))
       	mask = np.array(Image.open(os.path.join(lblpath,lblname)))
        lbl = np.array(np.array(Image.open(os.path.join(lblpath,lblname)))>0,dtype=np.uint8)
        box,size = get_area_rect(lbl)
        # imglbl = img*lbl
        imglbl = img
        img_top, img_left = box[1][0]-1, box[1][1]-1
        th, tw = size[0]+2,size[1]+2
        # 调整函数
        # if th == 65:
        #     print(name)
        #     th = 64

        imagecontainer = np.zeros((size[0], size[1], imglbl.shape[-1]), np.float32)
        imagecontainer = imglbl[img_top:img_top+th, img_left:img_left+tw]
        maskcontainer = np.zeros((size[0], size[1], imglbl.shape[-1]), np.float32)
        maskcontainer = mask[img_top:img_top+th, img_left:img_left+tw]

        scipy.misc.imsave(os.path.join(imagesavepath,lblname),imagecontainer)
        scipy.misc.imsave(os.path.join(masksavepath,lblname), maskcontainer)
        # print(name)


imgpath = "/media/jjchu/DataSets/spinalcord/test/image/"
lblpath = "/media/jjchu/seg/spinalcord/Results/UNet_GN2_new_model_batch12/testsite12_site12_centercrop128_lr0001_d25_loss2_grey_7layer_120_CP60/"
filepath = "/home/jjchu/MyResearch/spinalcord/Pytorch-UNet/data/testset_site12img_list.txt"
imagesavepath = "/media/jjchu/DataSets/spinalcord/test/cropimage12_rect/"
masksavepath = "/media/jjchu/DataSets/spinalcord/test/cropmask12_rect/"
crop_spinal_area_rect(imgpath,lblpath,filepath,imagesavepath,masksavepath)