import argparse
import os
from util import util
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # default = "voc2012"
        self.parser.add_argument('--dataroot', default="/home/jjchu/DataSet/VOCdevkit/VOC2012/", help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--netpath', type=str, default=1, help='weight for gradient penalty')

        self.parser.add_argument('--image_size', default=256, help='the size of the images')
        self.parser.add_argument('--cls_labels', default="/home/jjchu/My_Research/SegGAN/premodel/cls_labels.npy", help='path to images cls_labels.npy')
        self.parser.add_argument('--data_txt', default='trainval', help='train|val|trainval|trainall')
        self.parser.add_argument('--dataset_mode', type=str, default='segmentation', help='chooses how datasets are loaded. [unaligned | aligned | single | segmentation]')
        self.parser.add_argument('--c_dim', type=int, default=20, help='the num of the classes,basic 0| cgan 20')
        self.parser.add_argument('--conv_dim', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--cfg', default=[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], help='cfg')
        # self.parser.add_argument('--cfg', default=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], help='cfg')        
        self.parser.add_argument('--curr_dim', type=int, default=512, help='curr_dim')
        self.parser.add_argument('--n_blocks', type=int, default=3, help='the number of the dialted block')
        self.parser.add_argument('--block_channels', type=int, default=512, help='the number of the channels of the dialtedblock')
        
        self.parser.add_argument('--model', type=str, default='c_gan',
                                 help='chooses which model to use. cycle_gan, pix2pix, test,c_gan,gan')
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD,basic|vgg16')
        self.parser.add_argument('--which_model_netG', type=str, default='cgan', help='selects model to use for netG,basic|cgan|seggan')
        self.parser.add_argument('--D_repeat_num', type=int, default=8, help='only used if which_model_netD==n_layers')

        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        # self.parser.add_argument('--gpu_ids', type=str, default='2', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        
        self.parser.add_argument('--batchSize', type=int, default=44, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='this size to train')
        self.parser.add_argument('--cropSize', type=int, default=200, help='then crop to this size')

        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--resize_or_crop', type=str, default='scale_width_or_height_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|scale_width_or_height_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        self.parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{which_model_netG}_size{loadSize}')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()
        opt.isTrain = self.isTrain   # train or test

        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)

        # set gpu ids
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        args = vars(opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix
        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        self.opt = opt
        return self.opt
