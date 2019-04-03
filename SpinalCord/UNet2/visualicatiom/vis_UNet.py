import io
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import pdb
from unet import UNet_GN5
import os
import torch
import torch.nn as nn
from optparse import OptionParser
from data.spinal_cord_crop_dataset import spinalcordRealCropDataSet,spinalcordGen100DataSet
from PIL import Image
from torch import is_tensor
import matplotlib.image as mpimg
from sklearn.manifold import TSNE
import random
import matplotlib.pyplot as plt
from PIL import Image
import scipy.misc

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
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def saveFig(feature_conv,imgpath):
    feature_conv = tensor2im(feature_conv)
    # size_upsample = (256, 256)
    # feature_conv = cv2.resize(feature_conv, size_upsample)
    # cv2.imwrite(imgpath,feature_conv)
    image = Image.fromarray(feature_conv)
    image.save(imgpath)

def save_generate_mask(fake_img,fakepath):
    pred = fake_img.cpu().detach().numpy()
    pred = np.argmax(pred[0],0)
    save_mask = np.ceil(pred/2*255)
    scipy.misc.imsave(fakepath, save_mask)
    # save_mask = Image.fromarray(save_mask)
    # save_mask.save(fakepath)

def write_file(path,mess):
    print(mess)
    with open(path, 'a+') as net_file: # a+
        net_file.write(str(mess))
        net_file.write('\n')


def get_args():
    parser = OptionParser()
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',default=True, help='use cuda')
    parser.add_option('-s', '--scale', dest='scale', type='float',default=0.5, help='downscaling factor of the images')
    parser.add_option("--img_size", default=(100,100)) #(128，128)‘
    parser.add_option("--spinal_root", default="/home/jjchu/DataSet/spinalcord/", type=str)
    # parser.add_option("--spinal_root", default="/home/jjchu/Result/GANUNetResults/lblrescgan_center100_site12_lbl2img_gansite3_10PER_1GAN_20GMPER_resnet_5blocks_batch8_step200/test_latest/images/", type=str)
    parser.add_option("--snapshots", default="/home/jjchu/Result/UNetsnapshots/Real_ANTIALIAS_center100_site12_meanstd_imgs_b8_25l_1103/",type=str)
    parser.add_option("--savepath", default="/home/jjchu/Result/UNetResults/Real_ANTIALIAS_center100_site12_meanstd_imgs_b8_25l_1103/",type=str)  
    # parser.add_option("--train_list", default="./data/train_site12_list.txt", type=str)
    parser.add_option("--n_class", default=3, type=int)
    parser.add_option("--set", default="train",type=str)
    parser.add_option("--num_workers", default=4, type=int)
    parser.add_option("--nlabel", default=True)
    parser.add_option("--site", default=['site3','site3'])
    (options, args) = parser.parse_args()
    return options



def vis_features_layers(feature_conv):
    size_upsample = (100, 100)
    feature = np.max(feature_conv,axis=1) # (1,256,256)
    # feature = np.mean(feature_conv,axis=1)
    feature = np.squeeze(feature) #(256,256)
    feature = feature - np.min(feature)
    feature = feature/np.max(feature)
    feature = cv2.resize(feature, size_upsample)
    feature = np.uint8(255 * feature)
    heatmap = cv2.applyColorMap(feature, cv2.COLORMAP_JET)
    savepath = './mask/' + 'heatmap_down1_' + name[0]
    heatmap = Image.fromarray(heatmap)
    heatmap.save(savepath)


model_path = "/home/jjchu/Result/UNetsnapshots/real_100_100_site12_range11_imgs_b4_25l_1103/CP60.pth"
net = UNet_GN5(n_channels=3, n_classes=3)
net.eval()
net.cuda()
net.load_state_dict(torch.load(model_path))
args = get_args()
# for layer in layers:
features_blobs = []
net.down1.register_forward_hook(hook_feature)

# train_dataset = spinalcordGen100DataSet(args.spinal_root,img_size=args.img_size,site=['site3','site3'],batchsize=1, n_class=args.n_class, nlabel=True,set='train',real_or_fake='fake')
train_dataset = spinalcordRealCropDataSet(args.spinal_root,img_size=args.img_size,site=['site3','site3'],batchsize=1, n_class=args.n_class, nlabel=True,set='train')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
t = 120
for batch_idx, (data, target, label, size, name) in enumerate(train_loader):
    if batch_idx == t:
        image, target, label, size, name = data, target, label, size, name
        imgpath = './mask/' +'down1_'+ name[0]
        fakepath = './mask/' +'label_'+ name[0]
        break


saveFig(image,imgpath)

image = image.cuda()
label = net(image) # (1,3,256,256)
save_generate_mask(label,fakepath)

feature_conv = features_blobs[0] # (1,3,256,256)
MAPs = vis_features_layers(feature_conv)
