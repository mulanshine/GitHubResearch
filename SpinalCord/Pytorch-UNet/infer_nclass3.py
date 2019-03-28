import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
import scipy.misc
from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
from dice_loss import dice_coeff
from data.spinal_cord_crop_dataset import spinalcordCropDataSet
from data.spinal_cord_crop_dataset import spinalcordGenDataSet
from PIL import Image


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


def image_resize(pred,size):
    size = np.array(size)
    w,h = size[1],size[0]
    # print(w,h)
    # print(size)
    # print(pred.shape)
    if pred.shape != (h,w): # (64,48):
        pred = Image.fromarray(pred)
        pred = pred.resize((w,h)) # note (48,64) not (64,48)
        pred = np.asarray(pred, np.uint8)
        # print(pred.shape)
    return pred


def test_net_dice(args, net,batch_size=1,gpu=False):
    net.eval()
    # train_dataset = spinalcordCropDataSet(args.spinal_root,img_size=args.img_size,batchsize=args.batchsize, n_class=args.n_class, nlabel=args.nlabel,set=args.set)
    train_dataset = spinalcordGenDataSet(args.spinal_root,img_size=args.img_size,site=args.site,batchsize=args.batchsize, n_class=args.n_class, nlabel=args.nlabel,set=args.set)    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    # train_dataset1 = spinalcordCropDataSet(args.spinal_root,img_size=args.img_size,batchsize=args.batchsize, n_class=args.n_class, nlabel=args.nlabel,set=args.set)    
    # train_loader1 = torch.utils.data.DataLoader(train_dataset1, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    N_train = len(train_dataset)
    dice = np.array([0.0,0.0,0.0],dtype=np.float32)
    s = np.array([0.0,0.0,0.0],dtype=np.float32)
    for batch_idx, (data, target, label, size, name) in enumerate(train_loader):
        batch_idx = batch_idx + 1
        print(batch_idx)
        print(name)
        imgsavepath = os.path.join(args.savepath, name[0])

        imgs = np.array(data).astype(np.float32)
        true_masks = np.array(target).astype(np.float32)[0]

        imgs = torch.from_numpy(imgs)

        if gpu:
            imgs = imgs.cuda()

        masks_pred = net(imgs)
        masks_pred = masks_pred.cpu().detach().numpy()
        pred = np.argmax(masks_pred[0],0)
        # print(pred)
        pred_lbl = np.zeros((3,pred.shape[0],pred.shape[1]))
        for i in range(3):
            pred_lbl[i] = np.array((pred == i), dtype=np.uint8)

        for i in range(3):
            dice[i] = calculate_dice(pred_lbl[i],true_masks[i])

        s += dice
        save_mask = np.ceil(pred/2 *255)
        save_mask = image_resize(save_mask,size)
        # print(imgsavepath)
        scipy.misc.imsave(imgsavepath, save_mask)
        # break
    dice_result = s/N_train
    print(dice_result)
    return dice_result
    
def test_net(args, net,batch_size=1,gpu=False):
    net.eval()
    train_dataset = spinalcordCropDataSet(args.spinal_root,img_size=args.img_size,batchsize=args.batchsize, n_class=args.n_class, nlabel=args.nlabel,set=args.set)
    # train_dataset = spinalcordGenDataSet(args.spinal_root,img_size=args.img_size,site=args.site,batchsize=args.batchsize, n_class=args.n_class, nlabel=args.nlabel,set=args.set)    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    # train_dataset1 = spinalcordCropDataSet(args.spinal_root,img_size=args.img_size,batchsize=args.batchsize, n_class=args.n_class, nlabel=args.nlabel,set=args.set)    
    # train_loader1 = torch.utils.data.DataLoader(train_dataset1, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    N_train = len(train_dataset)
    for batch_idx, (data, target, label, size, name) in enumerate(train_loader):
        batch_idx = batch_idx + 1
        print(batch_idx)
        imgsavepath = os.path.join(args.savepath, name[0])

        imgs = np.array(data).astype(np.float32)
        imgs = torch.from_numpy(imgs)

        if gpu:
            imgs = imgs.cuda()

        masks_pred = net(imgs)
        masks_pred = masks_pred.cpu().detach().numpy()
        pred = np.argmax(masks_pred[0],0)
        # print(pred)
        save_mask = np.ceil(pred/2 *255)
        save_mask = image_resize(save_mask,size)
        print(imgsavepath)
        scipy.misc.imsave(imgsavepath, save_mask)


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=90, type='int',help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=1,type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.001,type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',default=True, help='use cuda')
    parser.add_option('-s', '--scale', dest='scale', type='float',default=0.5, help='downscaling factor of the images')
    parser.add_option("--img_size", default=(64,48))	# images
    parser.add_option("--spinal_root", default="/media/jjchu/seg/spinalcord/snapshots/CGAN/results/experiment_site24_Gres5block/test_latest/images/", type=str)
    # parser.add_option("--spinal_root", default="/media/jjchu/DataSets/spinalcord/", type=str)
    # parser.add_option('-c', '--load', default="/media/jjchu/seg/spinalcord/snapshots/UNet3/snapshots_site24_lr0001_b12_real64_48/CP30.pth",help='load file model')
    parser.add_option('-c', '--load', default="/media/jjchu/seg/spinalcord/snapshots/UNet3/snapshots_site24_lr0001_b12_gen64_48/CP150.pth",help='load file model')
    parser.add_option("--savepath", default="/media/jjchu/seg/spinalcord/Results/UNet3/snapshots_site24_lr0001_b12_gen64_48_CP/",type=str)
    # parser.add_option("--savepath", default="/media/jjchu/seg/spinalcord/Results/UNet3/snapshots_site24_lr0001_b12_real64_48",type=str)
    parser.add_option("--nlabel", default=True)
    parser.add_option("--n_class", default=3, type=int)
    parser.add_option("--set", default="test",type=str)
    parser.add_option("--num_workers", default=4, type=int)
    parser.add_option("--site", default=['site3','site3']) #,'site3'
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()
    
    net = UNet(n_channels=1, n_classes=3)
    # print(net)
    print(args.spinal_root)
    print(args.savepath)
    net.load_state_dict(torch.load(args.load))
    print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()

    print("test_net")

    dice = test_net_dice(args=args, net=net,batch_size=args.batchsize,gpu=args.gpu)
    print("#####################RESULT#######################")
    print(dice)
    print(args.load)
