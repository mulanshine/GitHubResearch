import os
import numpy as np 


filepath = "/home/jjchu/MyResearch/spinalcord/Pytorch-UNet/data/testset_site12img_list.txt"
imgpath = "/media/jjchu/DataSets/spinalcord/test/image/"

def write_filename_list(filepath,imgpath):
    imgList = os.listdir(imgpath)
    # lblList = os.listdir(lblpath)
    with open(filepath,"w+") as f:
        for imgname in imgList:
            if imgname.startswith("site1") or imgname.startswith("site2"):
                # site_sc_name = imgname[:10] # site1-sc03-mask-r1_1.pgm
                # for lblname in lblList:
                    # if lblname.startswith(site_sc_name) and lblname.split(".")[0].split("_")[-1]==imgname.split(".")[0].split("_")[-1]:
                writeline = imgname
                print(writeline)
                f.write(writeline+"\n")

write_filename_list(filepath,imgpath)