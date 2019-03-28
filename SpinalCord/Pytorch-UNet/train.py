import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from eval import eval_net
from unet import UNet
from unet import UNet_GN
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
from dice_loss import dice_coeff
from data.spinal_cord_dataset import spinalcordDataSet,resizeSpinalcordDataSet


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 25))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_net(args, net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=0.5):

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, str(save_cp), str(gpu)))

    # optimizer = optim.SGD(net.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)
    optimizer = optim.Adam(net.parameters(),lr=args.lr, betas=(args.beta1, args.beta2))
    # optimizer = optim.Adam(net.parameters())
    # criterion = nn.BCELoss()
    # criterion = dice_coeff()
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        line1 = 'Starting epoch {}/{}.\n'.format(epoch + 1, epochs)
        net.train()

        # # reset the generators
        # train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
        # val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)

        epoch_loss = 0

        train_dataset = resizeSpinalcordDataSet(args.spinal_root, args.train_list, img_size=args.img_size, crop_size=args.crop_size, resize_and_crop=args.resize_and_crop, batchsize=args.batchsize, n_class=args.n_class, nlabel=args.nlabel,set=args.set)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        # val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

        N_train = len(train_dataset)
        for batch_idx, (data, target, label, size, name) in enumerate(train_loader):
            batch_idx = batch_idx + 1
            imgs = np.array(data).astype(np.float32)
            true_masks = np.array(target).astype(np.float32)

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs)

            loss = dice_coeff(masks_pred, true_masks)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        adjust_learning_rate(optimizer,epoch)
              
        
        print('Epoch finished ! Loss: {}'.format(epoch_loss / batch_idx))
        line2 = 'Epoch finished ! Loss: {}\n'.format(epoch_loss / batch_idx)
        print('lr: {}'.format(optimizer.param_groups[0]['lr']))
        line3 = 'lr: {}\n'.format(optimizer.param_groups[0]['lr'])
          
        with open(args.logfile,'a+') as logf: 
            logf.write(line1)
            logf.write(line2)
            logf.write(line3)

        if save_cp and (epoch+1)%5 == 0:
            torch.save(net.state_dict(),
                       args.snapshots + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))
        # if 1:
        #     val_dice = 1 - eval_net(net, val_loader, gpu)
        #     print('Validation Dice Coeff: {}'.format(val_dice))


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=120, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=12,type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.001,type='float', help='learning rate')
    parser.add_option('--beta1', default=0.99,type='float', help='learning rate')
    parser.add_option('--beta2', default=0.999,type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',default=True, help='use cuda')
    parser.add_option('-c', '--load', default=None,help='load file model')
    # "/media/jjchu/seg/spinalcord/UNet/snapshots/CP40.pth"
    parser.add_option('-s', '--scale', dest='scale', type='float',default=0.5, help='downscaling factor of the images')
    parser.add_option("--train_list", default="/home/jjchu/MyResearch/spinalcord/Pytorch-UNet/data/train_site12_list.txt", type=str)
    parser.add_option("--crop_size", default=128, type=int)
    parser.add_option("--img_size", default=128, type=int)
    parser.add_option("--logfile", default="/media/jjchu/seg/spinalcord/snapshots/UNet_GN2_new_model_batch12/snapshots_site12_centercrop128_lr0001_d25_loss2_grey_7layer_120/log.txt",type=str)
    parser.add_option("--spinal_root", default="/media/jjchu/DataSets/spinalcord/", type=str)
    parser.add_option("--snapshots", default="/media/jjchu/seg/spinalcord/snapshots/UNet_GN2_new_model_batch12/snapshots_site12_centercrop128_lr0001_d25_loss2_grey_7layer_120/",type=str)
    parser.add_option("--unet", default="unet_gn",type=str)
    parser.add_option("--nlabel", default=True)
    parser.add_option("--n_class", default=2, type=int)
    parser.add_option("--set", default="train",type=str)
    parser.add_option("--num_workers", default=4, type=int)
    parser.add_option("--resize_and_crop", default='center_crop',type=str,help="center_crop|resize_and_center_crop|resize_and_random_crop|resize")
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()
    print(args)
    
    if args.unet == "unet":
        net = UNet(n_channels=1, n_classes=2)
    elif args.unet == "unet_gn":
        net = UNet_GN(n_channels=1, n_classes=2)
    print(net)
    if args.load != None:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        print("train_net")
        train_net(args=args, net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
