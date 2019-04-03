import sys
import os
from optparse import OptionParser
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from eval import eval_net
from unet import UNet_GN5,UNet_GN7,UNetTran_GN5,UNetTran_GN7
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
from dice_loss import DiceLoss, MulticlassDiceLoss
from data.spinal_cord_crop_dataset import spinalcordRealCropDataSet,spinalcordGen100DataSet,spinalcordGenlblresize2padataSet
import scipy
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
    pred = pred.resize(resize_size,Image.NEAREST)
    pred = np.array(pred,dtype=np.uint8)
    pred = encode_segmap(pred)
    return pred


def test_net_dice(args, net, batch_size=1, gpu=False,site=["site4","site4"]):
    net.eval()
    train_dataset = spinalcordRealCropDataSet(args.spinal_root,img_size=args.img_size,site=site,batchsize=1, n_class=args.n_class, nlabel=args.nlabel,set=args.set)

    # train_dataset = spinalcordGen100DataSet(args.spinal_root, img_size=args.img_size, site=site, batchsize=1, n_class=args.n_class,nlabel=args.nlabel,set=args.set,real_or_fake='real')
    # train_dataset = spinalcordGenlblresize2padataSet(args.spinal_root, img_size=args.img_size, site=site, batchsize=1, n_class=args.n_class,nlabel=args.nlabel,set=args.set,real_or_fake='real')
    # train_dataset = spinalcordCenterCropDataSet(args.spinal_root,img_size=args.img_size,site=site,batchsize=1, n_class=args.n_class, nlabel=args.nlabel,set=args.set)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    N_train = len(train_dataset)
    dice = np.array([0.0,0.0,0.0],dtype=np.float32)
    s = np.array([0.0,0.0,0.0],dtype=np.float32)
    for batch_idx, (data, target, label, size, name) in enumerate(train_loader):
        batch_idx = batch_idx + 1
        imgsavepath = os.path.join(args.savepath, name[0])
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
        label = image_resize(label,name)
        pred_lbl = np.zeros((3,pred.shape[0],pred.shape[1]))
        for i in range(3):
            pred_lbl[i] = np.array((pred == i), dtype=np.uint8)

        for i in range(3):
            dice[i] = calculate_dice(pred_lbl[i],true_masks[i])

        s += dice
        save_mask = np.ceil(pred/2*255)
        scipy.misc.imsave(imgsavepath, save_mask)

    dice_result = s/N_train
    print(site[0])
    print(dice_result)
    line4 = '{}:Dice result: {}\n'.format(site[0],dice_result[1])
          
    with open(args.logfile,'a+') as logf: 
        logf.write(line4)

    return dice_result


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 25))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_net(args, net,epochs=5,batch_size=1,lr=0.1,val_percent=0.05,save_cp=True,gpu=False):
    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, str(save_cp), str(gpu)))
    # lossdice = DiceLoss()
    lossdice = MulticlassDiceLoss()
    optimizer = optim.Adam(net.parameters(),lr=args.lr, betas=(args.beta1, args.beta2))

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        line1 = 'Starting epoch {}/{}.\n'.format(epoch + 1, epochs)
        net.train()

        epoch_loss = 0 
        # train_dataset = spinalcordGen100DataSet(args.spinal_root, img_size=args.img_size, site=args.site, batchsize=args.batchsize, n_class=args.n_class,nlabel=args.nlabel,set=args.set,real_or_fake=args.real_or_fake)
        # train_dataset = spinalcordresGen8064DataSet(args.spinal_root, img_size=args.img_size, site=args.site, batchsize=args.batchsize, n_class=args.n_class,nlabel=args.nlabel,set=args.set,real_or_fake=args.real_or_fake)
        train_dataset = spinalcordRealCropDataSet(args.spinal_root,img_size=args.img_size,site=args.site,batchsize=args.batchsize, n_class=args.n_class, nlabel=args.nlabel,set=args.set)
        # train_dataset = spinalcordCenterCropDataSet(args.spinal_root,img_size=args.img_size,site=args.site,batchsize=args.batchsize, n_class=args.n_class, nlabel=args.nlabel,set=args.set)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

        N_train = len(train_dataset)
        for batch_idx, (data, target, label, size, name) in enumerate(train_loader):

            # print(name)
            batch_idx = batch_idx + 1
            # print(batch_idx)
            imgs = np.array(data).astype(np.float32)
            true_masks = np.array(target).astype(np.float32)

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)
            if epoch == 0 and batch_idx == 1:
                print(imgs.shape)
                print(true_masks.shape)
                print(name)
            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs)

            # loss = dice_coeff(masks_pred, true_masks)
            loss = lossdice(masks_pred, true_masks,args.weight)

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        adjust_learning_rate(optimizer,epoch)  
        # print(epoch_loss)
        print(batch_idx)
        print('Epoch finished ! Loss: {}'.format(epoch_loss/batch_idx))

        line2 = 'Epoch finished ! Loss: {}\n'.format(epoch_loss / batch_idx)
        line3 = str(optimizer.param_groups[0]['lr'])
        print(line3)  
        with open(args.logfile,'a+') as logf: 
            logf.write(line1)
            logf.write(line2)
            logf.write(line3+'\n')

        if (epoch+1)%30==0:
            site = ['site1','site1']
            print("######################## test site1 ###############################")
            print(site[0])
            dice = test_net_dice(args=args, net=net, batch_size=1, gpu=args.gpu,site=site)
            print(dice)

        if (epoch+1)%10 == 0:
            site = ['site3','site3']
            print("######################## test site3 ###############################")
            dice = test_net_dice(args=args, net=net, batch_size=1, gpu=args.gpu,site=site)
            print(args.weight)

        if (epoch+1)%10 == 0:
            site = ['site4','site4']
            print("######################## test site4 ###############################")
            dice = test_net_dice(args=args, net=net, batch_size=1, gpu=args.gpu,site=site)
            print(args.weight)

        if save_cp and (epoch+1)%10 == 0:
            torch.save(net.state_dict(),
                       args.snapshots + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=120, type='int',help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=8,type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.001,type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',default=True, help='use cuda')
    parser.add_option('--beta1', default=0.99,type='float', help='learning rate')
    parser.add_option('--beta2', default=0.999,type='float', help='learning rate')
    parser.add_option('-s', '--scale', dest='scale', type='float',default=0.5, help='downscaling factor of the images')
    parser.add_option("--img_size", default=(100,100)) #(128，128)‘
    parser.add_option("--spinal_root", default="/home/jjchu/DataSet/spinalcord/", type=str)
    # parser.add_option("--spinal_root", default="/home/jjchu/Result/GANUNetResults/lblrescgan_center100_site12_lbl2img_gansite3_10PER_1GAN_20GMPER_resnet_5blocks_batch8_step200/test_latest/images/",type=str)
    # lblrescgan_TranUNet5_kernel7_center100_site12_lbl2img_gansite3_10PER_1GAN_20GMPER_resnet_5blocks_batch8_step200_real_fake_b12_25l_05103
    parser.add_option("--logfile", default="/home/jjchu/Result/UNetsnapshots/Real_kernel7_center100_site12_meanstd_imgs_b8_25l_05103_no/log.txt",type=str)
    parser.add_option("--snapshots", default="/home/jjchu/Result/UNetsnapshots/Real_kernel7_center100_site12_meanstd_imgs_b8_25l_05103_no/",type=str)
    parser.add_option("--savepath", default="/home/jjchu/Result/UNetResults/Real_kernel7_center100_site12_meanstd_imgs_b8_25l_05103_no/",type=str)  
    # parser.add_option('-c', '--load', default="/home/jjchu/Result/UNetsnapshots/Real_kernel7_center100_site12_meanstd_imgs_b8_25l_05103_no/CP50.pth",help='load file model')
    parser.add_option('-c', '--load', default=None,type=str)
    parser.add_option("--nlabel", default=True)
    parser.add_option("--n_class", default=3, type=int)
    parser.add_option("--set", default="train",type=str)
    # parser.add_option("--real_or_fake", default="real",type=str)
    parser.add_option("--num_workers", default=4, type=int)
    parser.add_option("--site", default=['site1','site2'])
    parser.add_option("--weight", default=[0.5,10.0,3.0])
    # parser.add_option("--resize_and_crop", default='center_crop',type=str,help="center_crop|resize_and_center_crop|resize_and_random_crop|resize")
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()
    print(args.snapshots)
    print(args.spinal_root)
    print(args.weight)
    # net = UNet_GN5(n_channels=3, n_classes=3)
    net = UNetTran_GN5(n_channels=3, n_classes=3)
    # print(net)
    if args.load != None and args.set == "test":
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    # if args.gpu:
    net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        if args.set == "train":
            print("train_net")
            train_net(args=args, net=net,epochs=args.epochs,batch_size=args.batchsize,lr=args.lr,gpu=args.gpu)
        elif args.set == "test":
            print("test_net")
            print(args.load)
            dice = test_net_dice(args=args, net=net, batch_size=1, gpu=args.gpu,site=['site3','site3'])
            dice = test_net_dice(args=args, net=net, batch_size=1, gpu=args.gpu,site=['site4','site4'])    

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
