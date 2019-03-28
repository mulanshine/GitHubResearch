import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class PixGANDDModel(BaseModel):
    def name(self):
        return 'PixGANDDModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        parser.set_defaults(DIMG_input_nc=3)
        parser.set_defaults(DAFF_input_nc=8)
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=10.0)
            parser.add_argument('--lambda_IMG', type=float, default=1.0)
            parser.add_argument('--lambda_AFF', type=float, default=1.0)
            parser.add_argument('--lambda_PER12', type=float, default=10.0)
            parser.add_argument('--lambda_PER34', type=float, default=5.0)
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        # self.loss_G = self.loss_G_GANIMG + self.loss_G_GANAFF + self.loss_G_L1 + self.loss_G_PER12 + self.loss_G_PER34
        self.loss_names = ['G_GANIMG','G_GANAFF', 'G_L1','G_PER12','G_PER34', 'D_IMG', 'DIMG_real','DIMG_fake','D_AFF', 'DAFF_real','DAFF_fake'] # 'G_AFF',,
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_ori12', 'real_ori34','fake12','fake34','target12']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D_IMG', 'D_AFF']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(opt.G_input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_IMG = networks.define_D(opt.DIMG_input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_AFF = networks.define_D(opt.DAFF_input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionPER = networks.perceptualLoss(perceptual_layers=3).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_DIMG = torch.optim.Adam(self.netD_IMG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_DAFF = torch.optim.Adam(self.netD_AFF.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_DIMG)
            self.optimizers.append(self.optimizer_DAFF)


    def set_input(self, input):
        self.real_ori12 = input['img12'].to(self.device)
        self.real_ori34 = input['img34'].to(self.device)
        self.target12 = input['target12'].to(self.device)
        self.real_aff12 = input['aff12'].to(self.device)
        self.image_paths = input['path']

    def forward(self):
        self.fake12 = self.netG(self.real_ori12)
        self.fake34 = self.netG(self.real_ori34)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D, loss_D_real, loss_D_fake

    def backward_D_IMG(self):
        fake34 = self.fake34
        self.loss_D_IMG, self.loss_DIMG_real, self.loss_DIMG_fake = self.backward_D_basic(self.netD_IMG, self.target12, fake34)

    def backward_D_AFF(self):
        fake_aff34 = networks.calculate_AFF(self.fake34)
        self.loss_D_AFF, self.loss_DAFF_real, self.loss_DAFF_fake = self.backward_D_basic(self.netD_AFF, self.real_aff12, fake_aff34)

    def backward_G(self):
        self.loss_G_L1 = self.criterionL1(self.fake12, self.target12) * self.opt.lambda_L1
        self.loss_G_GANIMG = self.criterionGAN(self.netD_IMG(self.fake34), True) * self.opt.lambda_IMG
        fake_aff34 = networks.calculate_AFF(self.fake34)
        self.loss_G_GANAFF = self.criterionGAN(self.netD_AFF(fake_aff34), True) * self.opt.lambda_AFF
        self.loss_G_PER12 = self.criterionPER(self.fake12, self.target12) * self.opt.lambda_PER12
        self.loss_G_PER34 = self.criterionPER(self.fake34, self.real_ori34) * self.opt.lambda_PER34
        self.loss_G = self.loss_G_GANIMG + self.loss_G_GANAFF + self.loss_G_L1 + self.loss_G_PER12 + self.loss_G_PER34
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_IMG, self.netD_AFF], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # D_A and D_B
        self.set_requires_grad([self.netD_IMG, self.netD_AFF], True)
        self.optimizer_DIMG.zero_grad()
        self.backward_D_IMG()
        self.optimizer_DIMG.step()
        self.optimizer_DAFF.zero_grad()
        self.backward_D_AFF()
        self.optimizer_DAFF.step()


















