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
from unet import UNet_GN5
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
from dice_loss import dice_coeff
from data.spinal_cord_crop_dataset import spinalcordGen200Normsite12DataSet,spinalcordCropDataSet, spinalcordGenlblresize2pad2NormdataSet, spinalcordGenlblresize2padataSet, spinalcordCenterCropDataSet, spinalcordGen200DataSet,spinalcordAnisodiffDataSet #,spinalcordGenlblresize2padatatestSet
from PIL import Image
import math


def encode_segmap(mask):
    valid_classes = [0, 128, 255]
    train_classes = [0, 1, 2]
    class_map = dict(zip(valid_classes, train_classes))
    for validc in valid_classes:
        mask[mask==validc] = class_map[validc]
    return mask

def calculate_dice(pred,mask):  
    pred = np.array(pred,dtype=np.uint8)
    mask = np.array(mask,dtype=np.uint8)
    w,h = pred.shape
    if pred.shape != mask.shape:
        # print("###########################")
        # print(pred.shape)
        # print(mask.shape)
        mask = Image.fromarray(np.array(mask,dtype=np.uint8))
        mask = mask.resize((h,w))
        mask = np.array(mask,dtype=np.uint8)
    pred = pred.reshape(-1,1)
    mask = mask.reshape(-1,1)
    eps = 0.0001
    inter = np.sum(pred*mask)
    union = np.sum(pred) + np.sum(mask) + eps
    dice = (2 * inter + eps) / union
    return dice

def image_resize(pred,name):
    name = name[0]
    size = np.array(pred).shape
    h,w = size[0],size[1]
    if name.startswith('site1') or name.startswith('site2'):
        resize_size = (math.floor(w/2.0),math.floor(h/2.0))
    elif name.startswith('site4'):
        resize_size = (math.floor(w/1.16),math.floor(h/1.16))
    elif name.startswith('site3'):
        resize_size = (w,h)
    
    pred = np.ceil(pred/2.0*255)
    pred = np.array(pred,dtype=np.uint8)
    pred = Image.fromarray(pred)
    pred = pred.resize(resize_size)
    pred = np.array(pred,dtype=np.uint8)
    pred = encode_segmap(pred)
    return pred


def test_net_dice(args, net,batch_size=1,gpu=False,k=0.9):
    net.eval()
    print(args.img_size[0])
    # train_dataset = spinalcordGenlblresize2pad2NormdataSet(args.spinal_root,img_size=args.img_size,site=args.site,batchsize=1, n_class=args.n_class, nlabel=args.nlabel,set=args.set,strends=args.real_or_fake)
    # # train_dataset = spinalcordCropDataSet(args.spinal_root,img_size=args.img_size,site=args.site,batchsize=args.batchsize, n_class=args.n_class, nlabel=args.nlabel,set=args.set)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # train_dataset = spinalcordCenterCropDataSet(args.spinal_root,img_size=args.img_size,site=args.site,batchsize=1, n_class=args.n_class, nlabel=args.nlabel,set=args.set)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    train_dataset = spinalcordGen200DataSet(args.spinal_root,img_size=args.img_size,site=args.site,batchsize=1, n_class=args.n_class, nlabel=args.nlabel, set=args.set,strends=args.real_or_fake)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)


    N_train = len(train_dataset)
    dice = np.array([0.0,0.0,0.0],dtype=np.float32)
    s = np.array([0.0,0.0,0.0],dtype=np.float32)
    NR3 = 0
    for batch_idx, (data, target, label, size, name) in enumerate(train_loader):
        batch_idx = batch_idx + 1
        imgsavepath = os.path.join(args.savepath, name[0])
        # print(name[0])
        imgs = np.array(data).astype(np.float32)
        true_masks = np.array(target).astype(np.float32)[0]
        imgs = torch.from_numpy(imgs)
        if gpu:
            imgs = imgs.cuda()

        masks_pred = net(imgs)
        masks_pred = masks_pred.cpu().detach().numpy()
        pred = np.argmax(masks_pred[0],0)
        pred = image_resize(pred,name)
        label = np.array(label,dtype=np.uint8)
        label = np.squeeze(label)
        pred_lbl = np.zeros((3,pred.shape[0],pred.shape[1]))
        for i in range(3):
            pred_lbl[i] = np.array((pred == i), dtype=np.uint8)
        
        for i in range(3):
            dice[i] = calculate_dice(pred_lbl[i],true_masks[i])

        s += dice
        save_mask = np.ceil(pred/2*255)
        scipy.misc.imsave(imgsavepath, save_mask)

    dice_result = s/N_train
    # dice_result = s/NR3
    print(args.site[0])
    print(args.real_or_fake)
    print(dice_result)
    return dice_result


def get_args():
    parser = OptionParser()
    parser.add_option('-b', '--batch-size', dest='batchsize', default=1,type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.001,type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',default=True, help='use cuda')
    parser.add_option('-s', '--scale', dest='scale', type='float',default=0.5, help='downscaling factor of the images')
    parser.add_option("--img_size", default=(200,200))	# images
    parser.add_option("--nlabel", default=True)
    parser.add_option("--n_class", default=3, type=int)
    parser.add_option("--set", default="test",type=str)
    parser.add_option("--num_workers", default=4, type=int)
    parser.add_option('-c', '--load', default="/share/jjchu/SpinalCord/UNetsnapshots/softmultiDiceLoss/segcgan_200_200_10L1_10PER_10SEG_resnet_5blocks_batch4_step400_relation_pad_meanstd_ddetach_fake_b12_25l_1103/CP90.pth",help='load file model')
    parser.add_option("--spinal_root", default="/share/jjchu/SpinalCord/CGANResults/segcgan_200_200_10L1_10PER_10SEG_resnet_5blocks_batch4_step400_relation_pad_meanstd_ddetach/test_latest/images/",type=str)
    parser.add_option("--savepath", default="/share/jjchu/SpinalCord/UNetResults/softmultiDiceLoss/test2/",type=str)  
 
    parser.add_option("--real_or_fake", default='fake',type=str)
    # parser.add_option("--real_or_fake", default='real',type=str)
    # parser.add_option("--site", default=['site3','site3']) #,'site3'
    parser.add_option("--site", default=['site3','site3']) #,'site3'
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()
    
    net = UNet_GN5(n_channels=3, n_classes=3)
    # print(net)
    print(args.spinal_root)
    print(args.savepath)
    net.load_state_dict(torch.load(args.load))
    print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()

    print("test_net")

    dice = test_net_dice(args=args, net=net,batch_size=args.batchsize,gpu=args.gpu,k=0)
    print("#####################RESULT#######################")
    print(dice)
    print(args.load)

