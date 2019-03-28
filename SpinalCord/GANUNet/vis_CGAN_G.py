import io
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import pdb
import os
import torch
import torch.nn as nn
from options.test_options import TestOptions
# from options.train_options import TrainOptions
from data import CreateDataLoader
from PIL import Image
from torch import is_tensor
import matplotlib.image as mpimg
import random
import matplotlib.pyplot as plt
from models import create_model


RANDOM_SEED = 1445754
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

def tensor2im(input_image, imtype=np.uint8):
    if is_tensor(input_image):
        image_tensor = input_image
    elif isinstance(input_image, Variable):
        image_tensor = input_image.data
    else:
        return input_image 
    image_numpy = image_tensor[0].cpu().detach().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def saveFig(feature_conv,imgpath):
    feature_conv = tensor2im(feature_conv)
    cv2.imwrite(imgpath,feature_conv)

def save_generate_fig(fake_img,fakepath):
    # feature_conv = torch.from_numpy(feature_conv)
    img = tensor2im(fake_img)
    # feature_conv = feature_conv.numpy()
    cv2.imwrite(fakepath,img)

def write_file(path,mess):
    print(mess)
    with open(path, 'a+') as net_file: # a+
        net_file.write(str(mess))
        net_file.write('\n')

def vis_features_layers(feature_conv,th=40,t=24,layers=18,set=False):
    size_upsample = (100, 100)
    feature = np.max(feature_conv,axis=1) # (1,256,256)
    # feature = np.mean(feature_conv,axis=1)
    feature = np.squeeze(feature) #(256,256)
    feature = (feature - np.min(feature))/(np.max(feature)- np.min(feature))
    # feature = feature.resize(size_upsample)
    print(feature.shape)
    feature = cv2.resize(feature, size_upsample)
    feature = np.uint8(255 * feature)
    # for x in range(len(feature[0])):
    #     for y in range(len(feature[1])):
    #         if feature[x,y] < th:
    #             feature[x,y] = 0
    #         elif set and feature[x,y] >= th:
    #             feature[x,y] = 255
    # feature = cv2.resize(feature, size_upsample)
    # output.append(cv2.resize(feature, size_upsample))
    heatmap = cv2.applyColorMap(feature, cv2.COLORMAP_JET)
    name = './test_feature/heat'+str(layer)+'.jpg'
    cv2.imwrite(name, heatmap)

opt = TestOptions().parse()
opt.num_threads = 1   # test code only supports num_threads = 1
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True    # no flip
opt.display_id = -1   # no visdom display
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
model.setup(opt)
model_path = "/home/jjchu/Result/GANUNetcheckpoints/lblrescgan_center100_site12_lbl2img_gansite3_10PER_1GAN_20GMPER_resnet_5blocks_batch8_step200/latest_net_G.pth"
# write_file(file_path, net)
model.eval()
model.netG.module.load_state_dict(torch.load(model_path))
# th = 0
t = 10
# t in range(1,10):
imgpath = './test_feature/' + str(t) + 'test.jpg'
targetpath = './test_feature/' + str(t) + 'target.jpg'
fakepath = './test_feature/' + str(t) + 'fake.jpg'
layer = 23 # 3,6,9,10,11-15,17,20,23
features_blobs = []
model.netG.module.model[layer].register_forward_hook(hook_feature)
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
for i, data in enumerate(dataset):
    if i == t:
        # 'real': imgRGB,'target':target, 'label':mask, 'path': imgpath
        # input_B = data['A'].float() # horse 2 zebra
        input_mask = data['label'] # zebra 2 horse
        input_real = data['real']
        input_target = data['target']
        break

saveFig(input_real,imgpath)
saveFig(input_target,targetpath)

input_mask = input_mask.cuda()
# image_paths = data['imgpath']  # match data['B'].float()
fake_img = model.netG(input_mask) # (1,3,256,256)
save_generate_fig(fake_img,fakepath)

feature_conv = features_blobs[0] # (1,3,256,256)
# img = cv2.imread(imgpath)
# height, width, _ = img.shape
MAPs = vis_features_layers(feature_conv,4,4,layer,False)



# MAPs = vis_heat_test(feature_conv,imgpath,th,t,layers,False)
# for th in [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]:
#     MAPs = vis_heat_test(feature_conv,imgpath,th,False)
    # MAPs = vis_heat_test(feature_conv,imgpath,th,True)
# # saveFig(feature_conv)
# # img_pil = Image.open(image_paths[0])
# # test_name = './result/result26/test25.jpg'
# # img_pil.save(test_name)
# # MAPs = vis_Threshold(feature_conv,th,True)
# # MAPs = vis_Threshold(feature_conv,th,False)
# MAPs = vis_heat_test(feature_conv,imgpath,0,False)
# # MAPs = vis_max(features_blobs[0])
# # MAPs = vis_mean(features_blobs[0])
# print(MAPs)
# # vis_channels(feature_conv)
# # img = mpimg.imread(image_paths[0])
# heatmap = cv2.applyColorMap(cv2.resize(MAPs[0],(width, height)), cv2.COLORMAP_JET)
# # # heatmap = cv2.applyColorMap(cv2.resize(MAPs[0],(width, height)), cv2.COLORMAP_AUTUMN)
# # cv2.imwrite('./result/result_th/heatmap_26.jpg', heatmap)
# # result = heatmap * 0.3 + img * 0.5
# # cv2.imwrite('./result/result_th/CAM_26.jpg', result)
# # heatname = './result_GAP/result/' + str(t) + 'heatmap_'+ str(layers) + '_' + str(th) +'.jpg'
# # camname = './result_GAP/result/' + str(t) + 'CAM_'+ str(layers) + '_' + str(th) +'.jpg'
# # heatname = './result_GMP/result_th/' + str(t) + 'heatmap_'+ str(layers) + '_' + str(th) + '.jpg'
# # camname = './result_GMP/result_th/' + str(t) + 'CAM_'+ str(layers) + '_' + str(th) + '.jpg'
# name = './result_GMP/test_heat_test.jpg'
# cv2.imwrite(name,heatmap)
# # result = heatmap * 0.3 + img * 0.5
# # cv2.imwrite(camname, result)




