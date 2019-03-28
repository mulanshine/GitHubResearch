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

from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
from dice_loss import dice_coeff
from data.spinal_cord_dataset import spinalcordDataSet,resizeSpinalcordDataSet

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
    h,w = size[0],size[1]
    if pred.shape != (h,w): # (64,48):
        pred = Image.fromarray(pred)
        pred = pred.resize((w,h)) # note (48,64) not (64,48)
        pred = np.asarray(pred, np.uint8)
    return pred

  
def test_net_dice(args, net, batch_size=1, gpu=False):   
    net.eval()
    # train_dataset = spinalcordDataSet(args.spinal_root, args.train_list, img_size=args.img_size, crop_size=args.crop_size, resize_and_crop=args.resize_and_crop, batchsize=args.batchsize, n_class=args.n_class, nlabel=args.nlabel,set=args.set)
    train_dataset = resizeSpinalcordDataSet(args.spinal_root, args.train_list, img_size=args.img_size, crop_size=args.crop_size, resize_and_crop=args.resize_and_crop, batchsize=args.batchsize, n_class=args.n_class, nlabel=args.nlabel,set=args.set)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dice = 0.0
    S1,S2,S3,S4 = 0.0,0.0,0.0,0.0
    N1,N2,N3,N4 = 0,0,0,0
    N_train = len(train_dataset)
    for batch_idx, (data, target,label,size,name) in enumerate(train_loader):
        print(batch_idx)
        batch_idx = batch_idx + 1
        imgsavepath = os.path.join(args.savepath, name[1][0])
        # imgsavepath = "/media/jjchu/seg/spinalcord/Results/predimg/test.jpg"
        imgs = np.array(data).astype(np.float32)
        true_mask = np.array(np.array(label)>0,dtype=np.uint8)
        imgs = torch.from_numpy(imgs)
        if gpu:
            imgs = imgs.cuda()

        pred_mask = net(imgs)
        pred_mask = pred_mask.cpu().detach().numpy()
        pred_mask = np.argmax(pred_mask[0],0)
        
        dice = calculate_dice(pred_mask,true_mask)
        pred_name = name[1][0]
        if pred_name.startswith('site1'):
            N1 += 1
            S1 += dice
        elif pred_name.startswith('site2'):
            N2 += 1
            S2 += dice
        if pred_name.startswith('site3'):
            N3 += 1
            S3 += dice
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
        imgsavepath = os.path.join(args.savepath, name[1][0])
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
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.001,type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',default=True, help='use cuda')
    parser.add_option('-s', '--scale', dest='scale', type='float',default=0.5, help='downscaling factor of the images')
    parser.add_option("--train_list", default="/home/jjchu/MyResearch/spinalcord/Pytorch-UNet/data/train_sites_list.txt", type=str)
    parser.add_option("--crop_size", default=128, type=int)
    parser.add_option("--img_size", default=128, type=int)
    parser.add_option("--spinal_root", default="/media/jjchu/DataSets/spinalcord/", type=str)
    # parser.add_option('-c', '--load', default="/media/jjchu/seg/spinalcord/snapshots/UNet2/snapshots_site13_centercrop128_lr0001_grey_150/CP50.pth",type=str)
    parser.add_option('-c', '--load', default="/media/jjchu/seg/spinalcord/snapshots/UNet2/snapshots_site13_resize025_centercrop128_lr001_grey_150/CP50.pth",type=str)
    # parser.add_option('-c', '--load', default="/media/jjchu/seg/spinalcord/snapshots/UNet2/snapshots_site13_centercrop128_lr001_grey_150/CP30.pth",type=str)
    parser.add_option("--savepath", default="/media/jjchu/seg/spinalcord/Results/UNet2/snapshots_site13_resize025_centercrop128_lr001_grey_150/",help='load file model')
    parser.add_option("--nlabel", default=True)
    parser.add_option("--n_class", default=2, type=int)
    parser.add_option("--set", default="test",type=str)
    parser.add_option("--num_workers", default=4, type=int)
    parser.add_option("--resize_and_crop", default='center_crop',type=str,help="center_crop|resize_and_center_crop|resize_and_random_crop|resize")
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=1, n_classes=2)
    # print(net)
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
