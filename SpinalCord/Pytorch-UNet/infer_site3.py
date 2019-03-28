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
from data.spinal_cord_dataset import spinalcordDataSet


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=10, type='int',help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=1,type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',default=True, help='use cuda')
    parser.add_option('-s', '--scale', dest='scale', type='float',default=0.5, help='downscaling factor of the images')
    parser.add_option("--train_list", default="/home/jjchu/MyResearch/spinalcord/Pytorch-UNet/data/test_site3_list.txt", type=str)
    parser.add_option("--crop_size", default=128, type=int)
    parser.add_option("--img_size", default=128, type=int)
    parser.add_option("--spinal_root", default="/media/jjchu/DataSets/spinalcord/", type=str)
    parser.add_option('-c', '--load', default="/media/jjchu/seg/spinalcord/snapshots/UNet2/snapshots_site24_centercrop128/CP50.pth",help='load file model')
    parser.add_option("--snapshots", default="/media/jjchu/seg/spinalcord/snapshots/UNet2/snapshots_site24_centercrop128/CP50.pth",type=str)
    parser.add_option("--savepath", default="/media/jjchu/seg/spinalcord/Results/UNet2/unet_test_site13_centercrop128_CP150/",type=str)
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
    model_path = "/media/jjchu/seg/spinalcord/snapshots/UNet2/snapshots_site24_centercrop128_grey/CP70.pth"
    net.load_state_dict(torch.load(model_path))
    print('Model loaded from {}'.format(args.load))
    if args.gpu:
        net.cuda()
    net.eval()

    test_path = "/media/jjchu/DataSets/spinalcord/train/image_site3_crop/"
    save_path = "/media/jjchu/DataSets/spinalcord/train/image_site3_save/"
    namelist = os.listdir(test_path)
    for name in namelist:
        spath = os.path.join(test_path,name)
        dpath = os.path.join(save_path,name)
        image = Image.open(spath).convert('L')
        image = np.array(image,dtype=np.float32)
        # normalize image
        image = image / 255.0
        image -= 0.5
        image = image/0.5
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        if args.gpu:
            image = image.cuda()
        masks_pred = net(image)
        masks_pred = masks_pred.cpu().detach().numpy()
        print(masks_pred.shape)
        mask = np.argmax(masks_pred[0],0)
        mask = mask *255
        print(mask.shape)
        scipy.misc.imsave(dpath, mask)