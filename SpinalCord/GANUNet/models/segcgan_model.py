import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class SegCGANModel(BaseModel):
    def name(self):
        return 'SegCGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='resnet_5blocks') #resnet_5blocks, resnet_3blocks_k3
        if is_train:
            parser.set_defaults(pool_size=0, no_lsgan=True)
            parser.set_defaults(lambda_L1=10.0) # ,lambda_AFF=0
            parser.set_defaults(lambda_PER=10.0)
            parser.set_defaults(lambda_SEG=10.0)
            parser.set_defaults(lambda_GAN=1.0)
            # parser.set_defaults(lambda_AFF=10.0)
            # parser.set_defaults(lambda_KL=10.0)
            parser.set_defaults(niter=100)
            parser.set_defaults(niter_decay=100)
            # parser.set_defaults(segmodel="/share/jjchu/SpinalCord/UNetsnapshots/softmultiDiceLoss/segcgan_denoise_norm2site12_site12_80_64_10L1_10PER_resnet_5blocks_batch4_step300_relation_pad_real_b12_25l_1103/CP60.pth")
            # parser.set_defaults(segmodel="/share/jjchu/SpinalCord/UNetsnapshots/softmultiDiceLoss/real_site12_anisodiff311_80_64_batch4_step400_b12_25l_1103/CP90.pth")
            parser.set_defaults(weight=[1.0,10.0,3.0])
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1','G_PER', 'G_SEG','G_AFF', 'D_real', 'D_fake'] # , 'G_SEG',
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real', 'fake']
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
            # self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionAFF = networks.AFFLoss().to(self.device)
            self.criterionPER3 = networks.perceptualLoss(perceptual_layers=3).to(self.device)
            self.criterionPER4 = networks.perceptualLoss(perceptual_layers=4).to(self.device)
            self.criterionPER5 = networks.perceptualLoss(perceptual_layers=5).to(self.device)
            # self.criterionSEG = networks.segLoss(model_path=opt.segmodel).to(self.device)
            # self.criterionKL = networks.DKLLoss().to(self.device)

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real = input['img'].to(self.device)
        self.aff = input['aff'].to(self.device)
        self.label = input['lbl'].to(self.device)
        self.image_paths = input['path']

    def forward(self):
        self.fake = self.netG(self.aff)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake = self.fake
        pred_fake = self.netD(fake.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)*self.opt.lambda_GAN

        # Real
        real = self.real
        pred_real = self.netD(real)
        self.loss_D_real = self.criterionGAN(pred_real, True)*self.opt.lambda_GAN

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake = self.fake
        pred_fake = self.netD(fake)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)*self.opt.lambda_GAN
        self.loss_G_AFF =self.criterionAFF(self.fake, self.aff) * self.opt.lambda_AFF

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake, self.real) * self.opt.lambda_L1

        # self.loss_G_PER = self.criterionPER(self.fake, self.real) * self.opt.lambda_PER
        self.loss_G_PER = (self.criterionPER3(self.fake, self.real) + self.criterionPER4(self.fake, self.real)+self.criterionPER5(self.fake, self.real)) * self.opt.lambda_PER/3.0

        self.loss_G_SEG = self.criterionSEG(self.fake, self.label, self.opt.weight) * self.opt.lambda_SEG
        # self.loss_G_KL = self.criterionKL(self.fake, self.real) * self.opt.lambda_KL
        # self.loss_G_SEG = 0
        self.loss_G = self.loss_G_GAN + self.loss_G_L1  + self.loss_G_PER + self.loss_G_AFF + self.loss_G_SEG # 
        # self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_PER + self.loss_G_SEG # + self.loss_G_KL
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