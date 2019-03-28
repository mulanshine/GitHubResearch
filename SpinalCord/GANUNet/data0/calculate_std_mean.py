import numpy as np 
from PIL import Image
import os

path = "/home/jjchu/MyResearch/Seg/spinalcord/CGAN/results/cgan_site12_80_64_L1_10_Gres5block_batch4_testsetep500_minmaxnorm/test_latest/images/"
nameList = os.listdir(path)
mean1,mean2,mean3,mean4 = 0.0,0.0,0.0,0.0
std1,std2,std3,std4 = 0.0,0.0,0.0,0.0
N1,N2,N3,N4 = 0,0,0,0,
for name in nameList:
	imgpath = os.path.join(path,name)
	img = np.array(Image.open(imgpath).convert('L'))
	img = img/255
	if name.endswith('_real.pgm'):
		if name.startswith('site1'):
			mean1 += np.mean(img)
			std1 += np.std(img)
			N1 += 1
		elif name.startswith('site2'):
			mean2 += np.mean(img)
			std2 += np.std(img)
			N2 += 1
		elif name.startswith('site3'):
			mean3 += np.mean(img)
			std3 += np.std(img)
			N3 += 1
		elif name.startswith('site4'):
			mean4 += np.mean(img)
			std4 += np.std(img)
			N4 += 1

mean1 = mean1/N1
mean2 = mean2/N2
mean3 = mean3/N3
mean4 = mean4/N4
std1 = std1/N1
std2 = std2/N2
std3 = std3/N3
std4 = std4/N4
print(f'N1:{N1}\nmean:{mean1},std:{std1}\nN2:{N2}:\n,mean:{mean2},std:{std2}\n,N3:{N3}:\nmean:{mean3},std:{std3}\n,N4:{N4}\nmean:{mean4},std:{std4}\n')