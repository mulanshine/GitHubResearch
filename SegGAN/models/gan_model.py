import torch
from torch.autograd import Variable
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F    

class GANModel(BaseModel):
    def name(self):
        return 'GANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.lambda_rec = opt.lambda_rec
        self.lambda_cls = opt.lambda_cls
        self.lambda_real = opt.lambda_real
        self.lambda_fake = opt.lambda_fake
        self.lambda_gp = opt.lambda_gp

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        # self.loss_names = ['D_real','D_fake', 'D_cls','G_fake','G_cls','G_rec']
        self.loss_names = ['D_real','D_fake', 'G_fake','G_rec']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_img', 'fake_img']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define netG_A、netG_B and netD_A、netD_B
        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X) c_dim=20, 
        self.netG = networks.define_G(opt.output_nc, opt.c_dim,opt.n_blocks, opt.block_channels, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.gpu_ids)
        if self.isTrain:
            self.netD = networks.define_D(opt.c_dim, opt.conv_dim, opt.curr_dim, opt.cfg, opt.image_size, opt.which_model_netD, opt.D_repeat_num, opt.norm, opt.init_type, opt.batch_norm, opt.gpu_ids)
        # define the loss and optimize function
        if self.isTrain:
            # self.fake_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionRec = torch.nn.L1Loss()
            self.criterionCls = networks.ClsLoss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        # if not isTrain, load the network
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

    # input data
    def set_input(self, input):
        # {'image':image,'paths':path,'label':label}
        input_img = input['image']
        input_label = input['label']
        if len(self.gpu_ids) > 0:
            input_img = input_img.cuda(self.gpu_ids[0], async=True)
            input_label = input_label.cuda(self.gpu_ids[0], async=True)

        self.input_img = input_img
        self.input_label = input_label
        self.image_paths = input['paths']
    
    def forward(self):
        self.real_img = Variable(self.input_img)
        self.real_label = Variable(self.input_label)
        # self.fake_img = self.netG(self.real_img, self.real_label)
        # self.pred_real, self.pred_real_cls = self.netD(self.real_img)
        # self.pred_fake, self.pred_fake_cls = self.netD(self.fake_img)

    def test(self):
        self.real_img = Variable(self.input_img, volatile=True)
        self.real_label = Variable(self.input_label, volatile=True)

    # def gradient_penalty(self, y, x):
        #     """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        #     weight = torch.ones(y.size()).to(self.device)
        #     dydx = torch.autograd.grad(outputs=y,
        #                                inputs=x,
        #                                grad_outputs=weight,
        #                                retain_graph=True,
        #                                create_graph=True,
        #                                only_inputs=True)[0]

        #     dydx = dydx.view(dydx.size(0), -1)
        #     dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        #     return torch.mean((dydx_l2norm-1)**2)

        # def classification_loss(self, logit, target):
        #     """Compute binary or softmax cross entropy loss."""
        #     return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)

    def backward_D(self):
        """
        self.pred_real、self.pred_real_cls
        """
        # Real
        self.pred_real, self.pred_real_cls = self.netD(self.real_img)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)
        # self.loss_D_cls = self.criterionCls(self.pred_real_cls, self.real_label)
        # Fake
        self.fake_img = self.netG(self.real_img)
        self.pred_fake, self.pred_fake_cls = self.netD(self.fake_img)
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)
        # combined loss
        # self.loss_D = self.lambda_real * self.loss_D_real + self.lambda_fake * self.loss_D_fake + self.lambda_cls * self.loss_D_cls 
        self.loss_D = self.lambda_real * self.loss_D_real + self.lambda_fake * self.loss_D_fake 

        self.loss_D.backward()
        # Compute loss for gradient penalty.
        # alpha = torch.rand(self.pred_real.size(0), 1, 1, 1).to(self.device)
        # self.grap_img = (alpha * self.pred_real.data + (1 - alpha) * self.pred_fake.data).requires_grad_(True)
        # self.pred_grap, _ = self.D(self.grap_img) # out_src, out_cls
        # self.loss_D_grap = self.gradient_penalty(self.pred_grap, self.grap_img)
        # self.loss_D = self.loss_D_real + self.loss_D_fake + self.lambda_cls * self.loss_D_cls + self.lambda_gp * self.loss_D_grap 

    def backward_G(self):
        self.fake_img = self.netG(self.real_img)
        self.pred_fake, self.pred_fake_cls = self.netD(self.fake_img)
        # self.pred_fake, self.pred_fake_cls = self.netD(self.fake_img.detach())

        self.loss_G_fake = self.criterionGAN(self.pred_fake, True)
        # self.loss_G_cls = self.criterionCls(self.pred_fake_cls, self.real_label)
        self.loss_G_rec = self.criterionRec(self.fake_img, self.real_img)
        # combined loss
        self.loss_G = self.loss_G_rec * self.lambda_rec + self.lambda_fake * self.loss_G_fake 
        # self.loss_G = self.loss_G_rec * self.lambda_rec + self.lambda_fake * self.loss_G_fake + self.lambda_cls * self.loss_G_cls
        
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    # def D_optimize_parameters(self):
    #     # forward
    #     self.forward()
    #     # D_A and D_B
    #     self.optimizer_D.zero_grad()
    #     self.backward_D()
    #     self.optimizer_D.step()


    # def G_optimize_parameters(self):
    #     # forward
    #     self.forward()
    #     # G_A and G_B
    #     self.optimizer_G.zero_grad()
    #     self.backward_G()
    #     self.optimizer_G.step()
