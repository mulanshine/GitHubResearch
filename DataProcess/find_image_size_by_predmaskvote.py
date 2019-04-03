import numpy as np 
from PIL import Image
import os
import scipy.misc

def vote_mask(imagespath,startstr):
    imgpaths = os.listdir(imagespath)
    for imgname in imgpaths:
        if imgname.startswith(startstr):
            imgpath = os.path.join(imagespath,imgname)
            images = np.array(Image.open(imgpath).convert('L'),dtype=np.float32)
            images = np.expand_dims(images,axis=0)


    for imgname in imgpaths:
        if imgname.startswith(startstr):
            imgpath = os.path.join(imagespath,imgname)
            img = np.array(Image.open(imgpath).convert('L'),dtype=np.float32)
            img = np.expand_dims(img,axis=0)
            images = np.concatenate((images, img), axis=0)

    images = images[1:]/255.0
    N = images.shape[0]
    M = int(N/3)
    image = np.sum(images,axis=0)
    image[image<M]=0
    image[image>=M]=1
    image = np.array(image,dtype=np.uint8)
    return image
				


# def get_min_area_rect(image):
#     # img = np.array(np.array(Image.open(impath))>0,dtype=np.uint8)
#     itemindex = np.argwhere(image == 1)
#     rect = cv2.minAreaRect(itemindex) 
#     box = cv2.boxPoints(rect) 
#     return box
 
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
    eh,ew = 100,100
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



#######################################
def crop_spinal_area_rect(imgpath,lblpath,maskpredpath,imagesavepath,masksavepath,logfile,startstr):#filepath,
    print(startstr)
    imgList = os.listdir(imgpath)
    maskList= os.listdir(lblpath)
    image = vote_mask(maskpredpath,startstr)
    box,size = get_area_rect(image)
    box,size = get_area_fixed_size(box,size)

    for imgname in imgList:
        if imgname.startswith(startstr):
            image = Image.open(os.path.join(imgpath,imgname)).convert('L')
            image = np.array(image, dtype=np.float32)
            img_top, img_left = box[1][0]-1, box[1][1]-1
            th, tw = size[0],size[1]
            # print(th,tw)
            imagecontainer = np.zeros((size[0], size[1]), np.float32)
            imagecontainer = image[img_top:img_top+th, img_left:img_left+tw]
            scipy.misc.imsave(os.path.join(imagesavepath,imgname),imagecontainer)

    for maskname in maskList:
        if maskname.startswith(startstr):
            mask = Image.open(os.path.join(lblpath,maskname)).convert('L')
            mask = np.array(mask, dtype=np.uint8)
            img_top, img_left = box[1][0]-1, box[1][1]-1
            th, tw = size[0],size[1]
            # print(th,tw)
            maskcontainer = np.zeros((size[0], size[1]), np.float32)
            maskcontainer = mask[img_top:img_top+th, img_left:img_left+tw]

            [[minX,maxY],[minX,minY],[maxX,minY],[maxX,maxY]] = box
            line = str(maskname)+' '+"[["+str(minX)+','+str(maxY)+"],["+str(minX)+','+str(minY)+"],["+str(maxX)+","+str(minY)+"],["+str(maxX)+','+str(maxY)+"]]\n"
            with open(logfile,'a+') as logf: 
                logf.write(line)
    
            scipy.misc.imsave(os.path.join(masksavepath,maskname), maskcontainer)

imgpath = "/home/jjchu/DataSet/spinalcord/centercrop_200/image_crop/"
lblpath = "/home/jjchu/DataSet/spinalcord/centercrop_200/mask_crop/"
maskpredpath="/home/jjchu/Result/UNet2Results/Real_centercrop200_nclass2_UNet5_kernel7_b8_18/"
imagespath = "/home/jjchu/DataSet/spinalcord/centercrop_200/image_crop/"
imagesavepath = "/home/jjchu/DataSet/spinalcord/predcrop_100/image_crop/"
masksavepath = "/home/jjchu/DataSet/spinalcord/predcrop_100/mask_crop/"
logfile = "/home/jjchu/DataSet/spinalcord/dataprocess/name_cropbox_from_centercrop200.txt"

for i in range(3,5):
    for j in range(1,11):
        if j <10:
            startstr = "site"+str(i)+'-sc0'+str(j)
        else:
            startstr = "site"+str(i)+'-sc'+str(j)

        crop_spinal_area_rect(imgpath,lblpath,maskpredpath,imagesavepath,masksavepath,logfile,startstr)