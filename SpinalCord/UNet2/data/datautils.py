import nibabel as nib
import numpy as np
import os
import scipy.misc
import shutil
import random
import numbers
import cv2
import scipy.misc

# SAVE NII.GZ to pgm images
# path = "/home/jjchu/Dataset/spinal_cord_segmentation/spinal_cord_segmentation_challenge/test/"
# spath = "/public/jjchu/Dataset/Spinel_cord_segmentation/test/"
def save_nii_slice_to_pgm(path,spath):
    nameList =os.listdir(path)
    for name in nameList:
        if os.path.splitext(name)[-1] == ".gz":
            img = nib.load(os.path.join(path,name))
            imgname = path.split('.')[0]
            img_arr = img.get_fdata()
            img_arr = np.squeeze(img_arr)
            img_arr = img_arr.transpose(2,0,1)
            sname = name.split(".")[0]
            for i in range(img_arr.shape[0]):
                savepath =os.path.join(spath,sname+"_"+str(i)+".pgm")
                print(savepath)
                img = img_arr[i]
                print(img.shape)
                scipy.misc.imsave(savepath, img)

# GET THE DELETE FILES' NAME LIST
# path = "/home/jjchu/Dataset/spinal_cord_segmentation/spinal_cord_segmentation_challenge/test/"
# pgmpath = "/public/jjchu/Dataset/Spinel_cord_segmentation/test/"
# deletenameList = delete_empty_file_by_leveltxt(path,pgmpath)
# deletenameList_arr = np.array(deletenameList)
# np.save("/home/jjchu/My_Research/spinal_cord_segmentation/data_process/test_delete_name.npy",deletenameList_arr)

# get the images and labels by the same site_sc
def get_site_sc_img_namelist(pgmpath,site_sc):
    List_site_sc = []
    nameList = os.listdir(pgmpath)
    for name in nameList:
        if site_sc in name:
            List_site_sc.append(name)
    return List_site_sc

# get the empty file's name by the index from the List_site_sc
def find_empty_file_by_index(i,List_site_sc):
    deleteList=[]
    for name in List_site_sc:
        if str(name.split(".")[0].split("_")[-1]) == str(i):
            deleteList.append(name)
    return deleteList


# find the empty file by the leveltxt
def delete_empty_file_by_leveltxt(path,pgmpath):
    List = os.listdir(path)
    deletenameList = []
    for name in List:
        site_sc = name[:10]
        List_site_sc = get_site_sc_img_namelist(pgmpath,site_sc)
        if os.path.splitext(name)[-1] == ".txt" and name.split(".")[0][:4] == "site":
            print(name)
            with open(os.path.join(path,name)) as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    line = lines[i]
                    if line.strip('\n').split(', ')[-1] == "-":
                        i = line.strip('\n').split(', ')[0]
                        deleteList = find_empty_file_by_index(i,List_site_sc)
                        print(deleteList)
                        deletenameList.extend(deleteList)
    return deletenameList

# move the deletefile from the files to the delete dir
# deletepath = "/public/jjchu/Dataset/Spinel_cord_segmentation/delete/test/"
# pgmpath = "/public/jjchu/Dataset/Spinel_cord_segmentation/test/"
# deleteListpath = "/home/jjchu/My_Research/spinal_cord_segmentation/data_process/test_delete_name.npy"
# deleteList = np.array(np.load(deleteListpath))
def move_deletefile_to_deletedir(pgmpath,deletepath,deleteList):
    for name in deleteList:
        srcpath = os.path.join(pgmpath, name)
        dstpath = os.path.join(deletepath, name) 
        print(srcpath)
        shutil.move(srcpath, dstpath)




# delete the error label caused by raters
# path = "/public/jjchu/Dataset/Spinel_cord_segmentation/test/"
# dpath = "/public/jjchu/Dataset/Spinel_cord_segmentation/delete_img_none/test/"
def delete_rater_error_label(path,dpath):
    List = os.listdir(path)
    for name in List:
        imgpath = os.path.join(path,name)
        img = Image.open(imgpath)
        img = np.array(img)
        num = np.sum(img)
        if num == 0:
            srcpath = os.path.join(path,name)
            dstpath = os.path.join(dpath,name)
            shutil.move(srcpath,dstpath)
            print(name)

# WRITE THE DILENAME LIST FOR TRAINING
# filepath = "/home/jjchu/My_Research/spinal_cord_segmentation/data/train_site12_list.txt"
# imgpath = "/public/jjchu/Dataset/Spinel_cord_segmentation/train/images/"
# lblpath = "/public/jjchu/Dataset/Spinel_cord_segmentation/train/mask/"
def write_filename_list(filepath,imgpath,lblpath):
    imgList = os.listdir(imgpath)
    lblList = os.listdir(lblpath)
    with open(filepath,"w+") as f:
        for imgname in imgList:
            if imgname.startswith("site1") or imgname.startswith("site2"):
                site_sc_name = imgname[:10] # site1-sc03-mask-r1_1.pgm
                for lblname in lblList:
                    if lblname.startswith(site_sc_name) and lblname.split(".")[0].split("_")[-1]==imgname.split(".")[0].split("_")[-1]:
                        writeline = imgname + ' ' + lblname
                        print(writeline)
                        f.write(writeline+"\n")


# return the classId in the dataset
def get_the_classlabel_for_img(path):
    d=set()
    img = Image.open(path)
    img = np.array(img)
    img = list(img.reshape(-1))
    for i in list(img):
        if i not in d:
            d.add(i)
    return d

def get_the_sc_imgname_List(filepath,imgpath):
    imgList = os.listdir(imgpath)
    with open(filepath,"w+") as f:
        for imgname in imgList:
            if imgname.startswith("site1") or imgname.startswith("site2"):
                f.write(imgname+"\n")




def RandomCrop(imgarr, cropsize):
    h, w, c = imgarr.shape

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space+1)
    else:
        cont_left = random.randrange(-w_space+1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space+1)
    else:
        cont_top = random.randrange(-h_space+1)
        img_top = 0

    container = np.zeros((cropsize, cropsize, imgarr.shape[-1]), np.float32)
    container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
        imgarr[img_top:img_top+ch, img_left:img_left+cw]

    return container


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
    size = [maxX - minX,maxY - minY]
    box = [[minX,maxY],[minX,minY],[maxX,minY],[maxX,maxY]]
    return box,size


# imgpath = "/media/jjchu/DataSets/spinalcord/train/images/"
# lblpath = "/media/jjchu/DataSets/spinalcord/train/mask/"
# filepath = "/home/jjchu/MyResearch/spinalcord/Pytorch-UNet/data/train_sites_list.txt"
# savepath = "/media/jjchu/DataSets/spinalcord/train/crop/"
def crop_spinal_area_rect(imgpath,lblpath,filepath,savepath):
    imgList = os.listdir(imgpath)
    lblList = os.listdir(lblpath)
    with open(filepath,"r") as f:
        nameList = f.readlines()

    for name in nameList:
        name = name.strip('\n')
        imgname, lblname = name.split(' ')
        img = np.array(Image.open(os.path.join(imgpath,imgname)))
        lbl = np.array(np.array(Image.open(os.path.join(lblpath,lblname)))>0,dtype=np.uint8)
        box,size = get_area_rect(lbl)
        imglbl = img*lbl
        img_top, img_left = box[1][0], box[1][1]
        th, tw = size[0],size[1]
        container = np.zeros((size[0], size[1], imglbl.shape[-1]), np.float32)
        container= imglbl[img_top:img_top+th, img_left:img_left+tw]

        scipy.misc.imsave(os.path.join(savepath,lblname),container)
