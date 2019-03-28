import sys
import os
from optparse import OptionParser
import numpy as np
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
import scipy.misc
import math
from eval import eval_net
from unet import UNet
from unet import UNet_GN7
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
from dice_loss import dice_coeff
from data.spinal_cord_dataset import spinalcordDataSet,resizeSpinalcordDataSet,infertestResizeSpinalcordDataSet

def calculate_dice(pred,mask):  
    pred = np.array(pred,dtype=np.uint8)
    mask = np.array(mask,dtype=np.uint8)
    pred = pred.reshape(-1,1)
    mask = mask.reshape(-1,1)
    eps = 0.0001
    inter = np.sum(pred*mask)
    union = np.sum(pred) + np.sum(mask) + eps
    dice = (2 * inter + eps) / union
    return dice


def image_resize(pred,size,name):
    name = name[0]
    h,w = size[0].float(),size[1].float()
    if name.startswith('site1') or name.startswith('site2'):
        pad_size = (math.ceil(2*w),math.ceil(2*h))
    elif name.startswith('site4'):
        pad_size = (math.ceil(1.16*w),math.ceil(1.16*h))
    elif name.startswith('site3'):
        pad_size = (math.ceil(w),math.ceil(h))
    hight = pad_size[1]
    weight = pad_size[0]
    pad_weight1 = int((weight - pred.shape[1])/2)
    pad_weight2 = weight - pred.shape[1] - pad_weight1
    pad_hight1 = int((hight - pred.shape[0])/2)
    pad_hight2 = hight - pred.shape[0] - pad_hight1

    pred = np.pad(pred,((pad_hight1, pad_hight2),(pad_weight1, pad_weight2)),"constant")
    pred = np.array(pred*255,dtype=np.uint8)
    pred = Image.fromarray(pred)
    pred = pred.resize((size[1],size[0]))
    pred = np.array(pred,dtype=np.uint8)
    return pred
  
def test_net_dice(args, net, batch_size=1, gpu=False):   
    net.eval()
    # train_dataset = spinalcordDataSet(args.spinal_root, args.train_list, img_size=args.img_size, crop_size=args.crop_size, resize_and_crop=args.resize_and_crop, batchsize=args.batchsize, n_class=args.n_class, nlabel=args.nlabel,set=args.set)
    train_dataset = infertestResizeSpinalcordDataSet(args.spinal_root, args.train_list, img_size=args.img_size, crop_size=args.crop_size, resize_and_crop=args.resize_and_crop, batchsize=args.batchsize, n_class=args.n_class, nlabel=args.nlabel,set=args.set)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dice = 0.0
    S1,S2,S3,S4 = 0.0,0.0,0.0,0.0
    N1,N2,N3,N4 = 0,0,0,0
    NN = 0
    N_train = len(train_dataset)
    for batch_idx, (data, target,label,size,name) in enumerate(train_loader):
#         torch.Size([1, 1, 128, 128])
# torch.Size([1, 2, 654, 774])
# [tensor([654]), tensor([774])]
# [['site3-sc05-image_23.pgm'], ['site3-sc05-mask-r3_23.pgm']]
        # print(batch_idx)
        batch_idx = batch_idx + 1
        print(name)
        imgsavepath = os.path.join(args.savepath, name[0])
        # imgsavepath = "/media/jjchu/seg/spinalcord/Results/predimg/test.jpg"
        imgs = np.array(data).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        if gpu:
            imgs = imgs.cuda()

        pred_mask = net(imgs)
        pred_mask = pred_mask.cpu().detach().numpy()
        pred_mask = np.argmax(pred_mask[0],0)
        pred_mask = image_resize(pred_mask,size,name)
        true_mask = np.array(np.array(label)>0,dtype=np.uint8)
        pred_mask = np.array(np.array(pred_mask)>0,dtype=np.uint8)
        dice = calculate_dice(pred_mask,true_mask[0][0])
        pred_name = name[0]
        
        if dice < 0.7:
            print(dice)
            print(pred_name)
            NN += 1
        if pred_name.startswith('site1'):
            N1 += 1
            S1 += dice
        elif pred_name.startswith('site2'):
            N2 += 1
            S2 += dice
        if pred_name.startswith('site3'):
            N3 += 1
            S3 += dice
            # print(dice)
        if pred_name.startswith('site4'):
            N4 += 1
            S4 += dice
        pred_mask = pred_mask *255
        scipy.misc.imsave(imgsavepath, pred_mask)

    dice1 = S1/N1
    dice2 = S2/N2
    dice3 = S3/N3
    dice4 = S4/N4
    print(N1,dice1,N2,dice2,N3,dice3,N4,dice4)
    print(NN)
    
    return N1,dice1,N2,dice2,N3,dice3,N4,dice4      
# '''

def test_net(args, net, batch_size=1, gpu=False):   
    net.eval()
    # train_dataset = spinalcordDataSet(args.spinal_root, args.train_list, img_size=args.img_size, crop_size=args.crop_size, resize_and_crop=args.resize_and_crop, batchsize=args.batchsize, n_class=args.n_class, nlabel=args.nlabel,set=args.set)
    train_dataset = resizeSpinalcordDataSet(args.spinal_root, args.train_list, img_size=args.img_size, crop_size=args.crop_size, resize_and_crop=args.resize_and_crop, batchsize=args.batchsize, n_class=args.n_class, nlabel=args.nlabel,set=args.set)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    N_train = len(train_dataset)
    for batch_idx, (data, target, size, name) in enumerate(train_loader):
        batch_idx = batch_idx + 1
        imgsavepath = os.path.join(args.savepath, name[0])
        # imgsavepath = "/media/jjchu/seg/spinalcord/Results/predimg/test.jpg"
        imgs = np.array(data).astype(np.float32)
        # true_masks = np.array(target).astype(np.float32)

        imgs = torch.from_numpy(imgs)
        # true_masks = torch.from_numpy(true_masks)

        if gpu:
            imgs = imgs.cuda()
            # true_masks = true_masks.cuda()

        masks_pred = net(imgs)
        masks_pred = masks_pred.cpu().detach().numpy()
        # print(masks_pred.shape)

        mask = np.argmax(masks_pred[0],0)
        mask = mask *255
        # print(mask.shape)
        scipy.misc.imsave(imgsavepath, mask)

def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=10, type='int',help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=1,type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.01,type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',default=True, help='use cuda')
    parser.add_option('-s', '--scale', dest='scale', type='float',default=0.5, help='downscaling factor of the images')
    parser.add_option("--train_list", default="/home/jjchu/MyResearch/spinalcord/Pytorch-UNet/data/testset_site12img_list.txt", type=str)
    parser.add_option("--crop_size", default=128, type=int)
    parser.add_option("--img_size", default=128, type=int)
    parser.add_option("--spinal_root", default="/media/jjchu/DataSets/spinalcord/", type=str)
    # parser.add_option('-c', '--load', default="/media/jjchu/seg/spinalcord/snapshots/UNet2/snapshots_site13_centercrop128_lr0001_grey_150/CP50.pth",type=str)
    parser.add_option('-c', '--load', default="/media/jjchu/seg/spinalcord/snapshots/UNet_GN2_new_model_batch12/snapshots_site12_centercrop128_lr0001_d25_loss2_grey_7layer_120/CP60.pth",type=str)
    # parser.add_option('-c', '--load', default="/media/jjchu/seg/spinalcord/snapshots/UNet2/snapshots_site13_centercrop128_lr001_grey_150/CP30.pth",type=str)
    parser.add_option("--savepath", default="/media/jjchu/seg/spinalcord/Results/UNet_GN2_new_model_batch12/testsite12_site12_centercrop128_lr0001_d25_loss2_grey_7layer_120_CP60/",help='load file model')
    parser.add_option("--nlabel", default=True)
    parser.add_option("--n_class", default=2, type=int)
    parser.add_option("--unet", default="unet_gn",type=str)
    parser.add_option("--set", default="test",type=str)
    parser.add_option("--num_workers", default=4, type=int)
    parser.add_option("--resize_and_crop", default='center_crop',type=str,help="center_crop|resize_and_center_crop|resize_and_random_crop|resize")
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()
    if args.unet == "unet":
        net = UNet(n_channels=1, n_classes=2)
    elif args.unet == "unet_gn":
        net = UNet_GN7(n_channels=1, n_classes=2)
    # net = UNet(n_chansnels=1, n_classes=2)
    net.load_state_dict(torch.load(args.load))
    print('Model loaded from {}'.format(args.load))
    if args.gpu:
        net.cuda()
    try:
        print("test_net")
        N1,dice1,N2,dice2,N3,dice3,N4,dice4 = test_net_dice(args=args, net=net,
                  batch_size=args.batchsize,
                  gpu=args.gpu)
        print(N1,dice1,N2,dice2,N3,dice3,N4,dice4)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    print(args.load)
