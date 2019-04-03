import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class LblCGANModel(BaseModel):
    def name(self):
        return 'LblCGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='resnet_5blocks') #resnet_5blocks, resnet_3blocks_k3
        parser.set_defaults(niter=100)
        parser.set_defaults(niter_decay=100)
        if is_train:
            parser.set_defaults(lambda_L1=10.0) # ,lambda_AFF=0
            parser.set_defaults(lambda_PER=10.0)
            parser.set_defaults(lambda_GMPER=20.0)
            # parser.set_defaults(lambda_GWMPER=10.0)
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G','G_GAN', 'G_PER','G_GMPER','G_L1bk', 'D','D_target', 'D_fake'] # 'G_AFF','G_PER','G_L1', 'G_GWMPER',
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if self.isTrain:
            self.visual_names = ['real','fake','target'] # 'normreal',
        else:
            self.visual_names = ['real','fake']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(opt.G_input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.D_input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            # self.criterionL1 = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionGML1 = torch.nn.L1Loss()
            self.criterionGMPER = networks.perceptualLoss(perceptual_layers=3).to(self.device)
            self.criterionPER = networks.perceptualLoss(perceptual_layers=3).to(self.device)

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        # self.normreal = input['img'].to(self.device)
        # self.aff = input['aff'].to(self.device)
        self.target = input['target'].to(self.device)
        self.real = input['real'].to(self.device)
        self.label = input['label'].to(self.device)
        self.image_paths = input['path']

    def forward(self):
        self.fake = self.netG(self.label)

    def backward_D(self):
        # Fake
        fake = self.fake
        pred_fake = self.netD(fake.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        # real = self.real
        # pred_real = self.netD(real)
        # self.loss_D_real = self.criterionGAN(pred_real, True)

        # target
        target = self.target
        pred_target = self.netD(target)
        self.loss_D_target = self.criterionGAN(pred_target, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_target) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        fake = self.fake
        pred_fake = self.netD(fake)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # L1_background
        real_bg = self.real * (self.label[0])
        fake_bg = fake * (self.label[0])
        self.loss_G_L1bk = self.criterionL1(fake_bg, real_bg) * self.opt.lambda_L1

        fake_gm = fake * (self.label[1]+self.label[2])
        real_gm = self.real * (self.label[1]+self.label[2])
        # fake_gwm = fake * (self.label[1] + self.label[2])
        # real_gwm = self.real * (self.label[1] + self.label[2])

        self.loss_G_GMPER = self.criterionGMPER(fake_gm,real_gm) * self.opt.lambda_GMPER
        # self.loss_G_GWMPER = self.criterionGMPER(fake_gwm,real_gwm) * self.opt.lambda_GWMPER
        self.loss_G_PER = self.criterionPER(self.fake, self.real) * self.opt.lambda_PER

        self.loss_G = self.loss_G_GAN + self.loss_G_PER + self.loss_G_GMPER + self.loss_G_L1bk# + self.loss_G_GWMPER # + self.loss_G_L1 
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # update G
        # self.forward()
        # self.set_requires_grad(self.netD, False)
        # self.optimizer_G.zero_grad()
        # self.backward_G()
        # self.optimizer_G.step()
