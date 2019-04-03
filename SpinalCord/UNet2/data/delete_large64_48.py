import numpy as np
import os
from PIL import Image
import shutil

imgspath = "/media/jjchu/DataSets/spinalcord/train/cropimage/"
maskspath = "/media/jjchu/DataSets/spinalcord/train/cropmask/"
maskdpath = "/media/jjchu/DataSets/spinalcord/train/delete_crop/cropdeletemask/"
imgdpath = "/media/jjchu/DataSets/spinalcord/train/delete_crop/cropdeleteimg/"
imgnamelist = os.listdir(imgspath)
masknamelist = os.listdir(maskspath)
for name1 in imgnamelist:
	spath1 = os.path.join(imgspath,name1)
	dpath1 = os.path.join(imgdpath,name1)
	img = np.array(Image.open(os.path.join(imgspath,name1)))
	h,w = img.shape
	if h > 64 or w > 48:
		shutil.move(spath1,dpath1)

for name2 in masknamelist:
	spath2 = os.path.join(maskspath,name2)
	dpath2 = os.path.join(maskdpath,name2)	
	mask = np.array(Image.open(os.path.join(maskspath,name2)))
	h,w = mask.shape
	if h > 64 or w > 48:
		shutil.move(spath2,dpath2)

