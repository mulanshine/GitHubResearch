import numpy as np 
import os
from PIL import Image
import cv2
import scipy.misc
import math
from datautils import RandomCrop, CenterCrop

def centercrop_spinal_cord(imgpath,lblpath,filepath,imagesavepath,masksavepath,crop_size):
    imgList = os.listdir(imgpath)
    lblList = os.listdir(lblpath)
    # 'site3-sc05-image_23.pgm site3-sc05-mask-r3_23.pgm\n'
    with open(filepath,"r") as f:
        nameList = f.readlines()

    for name in nameList:
        name = name.strip('\n')
        imgname, lblname = name.split(' ')
        image = Image.open(os.path.join(imgpath,imgname))
       	mask = Image.open(os.path.join(lblpath,lblname))
        size = np.array(image, dtype=np.uint32).shape
        mask = np.array(mask,dtype=np.uint8)
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

        image = np.expand_dims(image, axis=2)
        mask = np.expand_dims(mask, axis=2) 
        img_label = np.concatenate((image, mask), axis=-1)
        img_label = CenterCrop(img_label,crop_size)
        image = img_label[..., 0]
        label = img_label[..., 1]
        label = np.squeeze(label)
        image = np.squeeze(image)

        scipy.misc.imsave(os.path.join(imagesavepath,lblname),image)
        scipy.misc.imsave(os.path.join(masksavepath,lblname), label)
        print(lblname)



imgpath = "/home/jjchu/DataSet/spinalcord/images/"
lblpath = "/home/jjchu/DataSet/spinalcord/mask/"
filepath = "/home/jjchu/GitHubResearch/SpinalCord/Pytorch-UNet/data/train_site34_list.txt"
imagesavepath = "/home/jjchu/DataSet/spinalcord/centercrop_200/image_crop/"
masksavepath = "/home/jjchu/DataSet/spinalcord/centercrop_200/mask_crop/"
crop_size = 200
centercrop_spinal_cord(imgpath,lblpath,filepath,imagesavepath,masksavepath,crop_size)

