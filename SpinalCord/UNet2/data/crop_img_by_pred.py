import numpy as np 
import os
from PIL import Image
import scipy.misc

# "/media/jjchu/DataSets/spinalcord/train/images/site1-sc04-image_1.pgm"
imgpath = "/media/jjchu/DataSets/spinalcord/train/images/"
maskpath = "/media/jjchu/DataSets/spinalcord/train/mask/"
predpath = "/media/jjchu/seg/spinalcord/Results/unettest_center_crop_100/"
savepath = "/media/jjchu/seg/spinalcord/Results/predimg/"
# site2-sc06-mask-r3_5.pgm
namelist = os.listdir(predpath)
namelist = ["site2-sc08-mask-r4_2.pgm"]
for name in namelist:
	print(name)
	pred = np.array(np.array(Image.open(os.path.join(predpath,name)))>0)
	mask = np.array(np.array(Image.open(os.path.join(maskpath,name)))>0)
	imgname = name[:11] +'image_' + name.split('_')[-1]
	image = np.array(Image.open(os.path.join(imgpath,imgname)))
	res_pred = pred * image
	res_mask = mask * image

	scipy.misc.imsave(os.path.join(savepath,'resp_'+imgname), res_pred)
	scipy.misc.imsave(os.path.join(savepath,'resm_'+imgname), res_mask)
	scipy.misc.imsave(os.path.join(savepath,'pred_' + name), pred*255)
	scipy.misc.imsave(os.path.join(savepath,imgname), image)
	scipy.misc.imsave(os.path.join(savepath,'mask_' + name), mask*255)

	print(name)
	break