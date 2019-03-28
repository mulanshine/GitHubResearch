import numpy as np 
from PIL import Image
import os

imgspath = "/home/jjchu/MyResearch/Seg/spinalcord/CGAN/results/cgan_site12_80_64_L1_10_Gres5block_batch4_testsetep500_minmaxnorm/test_latest/images/"
# maskspath = '/home/jjchu/MyResearch/Seg/spinalcord/Pytorch-UNet/results/CrossEntropyLoss/snapshots_site12_lr0001_b12_25l_gen80_64_1103/'
maskspath = "/home/jjchu/MyResearch/Seg/spinalcord/Pytorch-UNet/results/CrossEntropyLoss/snapshots_site12_lr0001_b12_25l_crop80_64_w1103/"
spath = "/home/jjchu/MyResearch/Seg/spinalcord/Pytorch-UNet/analysisresult/imgmaskcrop/"
L = os.listdir(imgspath)
namelist=[]
for name in L:
	if name.startswith('site3') and name.endswith('_fake.pgm'):
		namelist.append(name)

for name in namelist:
	maskname = '_'.join((name.split('_')[0:-1]))+'.pgm'
	imgpath = os.path.join(imgspath,name)
	maskpath = os.path.join(maskspath,maskname)
	savepath = os.path.join(spath,name)
	img = np.array(Image.open(imgpath).convert('L'),dtype=np.uint8)
	mask = np.array(Image.open(maskpath).convert('L'),dtype=np.uint8)
	mask[mask != 128] = 1
	mask[mask == 128] = 0
	mask1 = np.zeros((mask.shape[0],mask.shape[1]))
	for i in range(mask.shape[0]):
		for j in range(1,mask.shape[1]-1):
			if mask[i][j-1] != mask[i][j] or mask[i][j] != mask[i][j+1]:
				mask1[i][j]=128
			else:
				mask1[i][j]=0
	# mask2 = Image.fromarray(np.array(mask1,dtype=np.uint8))
	# mask2.save(os.path.join(spath,'mask1.pgm'))
	# print(mask.sum())
	imgmask = img + mask1*2
	imgmask[imgmask>255] = 255
	imgmask = Image.fromarray(np.array(imgmask,dtype=np.uint8))
	imgmask.save(savepath)
	# break




