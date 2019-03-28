import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
import torch.nn.functional as F 
import os
import os.path as osp

###############################################################################
# Helper Functions
###############################################################################
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def write_net_file(opt,model):
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    netD_path = os.path.join(expr_dir, 'netD.txt')
    netG_path = os.path.join(expr_dir, 'netG.txt')
    with open(netD_path, 'w') as net_D: # a+
        net_D.write(str(model.netD))
        net_D.write('\n')
    with open(netG_path, 'w') as net_G: # a+
        net_G.write(str(model.netG))
        net_G.write('\n')

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net

# def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
  

def define_G(output_nc=3, c_dim=20, n_blocks=3, block_channels=512, which_model_netG='basic', norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)
    if which_model_netG == 'basic':
        netG = NetGenerator(output_nc=3, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks, block_channels=block_channels, padding_type='reflect')
    elif which_model_netG == 'cgan':
        netG = NetCGenerator(output_nc=3, c_dim=20, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks, block_channels=block_channels, padding_type='reflect')
    elif which_model_netG == 'seggan':
        netG = SegGenerator(output_nc=23, c_dim=20, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks, block_channels=block_channels, padding_type='reflect')
    # elif which_model_netG == 'unet_128':
    #     netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    # else:
    #     raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, gpu_ids)


def define_D(c_dim, conv_dim,curr_dim, cfg, image_size, which_model_netD, repeat_num=6, norm='batch', init_type='normal', batch_norm=True, gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(image_size, conv_dim, c_dim,repeat_num)
    elif which_model_netD == 'vgg16':
        netD = VGGDiscriminator(image_size, cfg, curr_dim, c_dim, norm_layer, batch_norm=True)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % which_model_netD)
    return init_net(netD, init_type, gpu_ids)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


##############################################################################
# Classes
##############################################################################

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the x
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label  # 1.0
        self.fake_label = target_fake_label  # 0.0
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label) 
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label) 
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        # return self.loss(input, target_tensor)/(input.size(2)*input.size(1))
        return self.loss(input, target_tensor)

# define the loss of classfication
class ClsLoss(nn.Module):
    def __init__(self, tensor=torch.FloatTensor):
        super(ClsLoss, self).__init__()
        self.real_label_var = None
        self.pred_label_var = None
        self.Tensor = tensor
        # self.loss = F.binary_cross_entropy_with_logits()

    def get_target_tensor(self, pred_label, real_label):
        target_label = None
        create_label = ((self.real_label_var is None) or
                        (self.real_label_var.numel() != pred_label.numel()))
        com_label = (pred_label.numel() != real_label.numel())
        if create_label:
            if com_label:
                real_label = real_label.view(real_label.size(0), real_label.size(1), 1, 1)
                real_label = real_label.repeat(1, 1, pred_label.size(2), pred_label.size(3))
                # real_tensor = self.Tensor(pred_label.size()).fill_(self.real_label) 
                self.real_label_var = real_label
            else:
                self.real_label_var = real_label
        target_label = self.real_label_var
        return target_label
        
    def __call__(self, pred_label, real_label):
        target_label = self.get_target_tensor(pred_label, real_label)
        # target_label = target_label.type(torch.cuda.LongTensor)
        # return F.cross_entropy(pred_label, target_label)
        # /20 : because it has 20 categories
        return F.binary_cross_entropy_with_logits(pred_label, target_label, size_average=False) / (pred_label.size(0))
        # return F.binary_cross_entropy_with_logits(pred_label, target_label) 
        # return nn.BCEWithLogitsLoss(pred_label, target_label)
# return out_src, out_cls
class NLayerDiscriminator(nn.Module):
    """NLayerDiscriminator network with PatchGAN."""
    def __init__(self, image_size=256, conv_dim=64, c_dim=20, repeat_num=8):
        super(NLayerDiscriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            if i < 4:
                layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
                layers.append(nn.BatchNorm2d(curr_dim*2))
                layers.append(nn.LeakyReLU(0.2, True))
                curr_dim = curr_dim * 2
            else:
                layers.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=4, stride=2, padding=1))
                layers.append(nn.BatchNorm2d(curr_dim))
                layers.append(nn.LeakyReLU(0.2, True))

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()

    def forward(self, x):
        h = self.main(x)
        out_src = self.sigmoid(self.conv1(h))
        out_cls = self.sigmoid(self.conv2(h)) # 1*20*1*1
        # out_cls = out_cls.view(out_cls.size(0), out_cls.size(1))
        # out_cls = self.softmax(out_cls) 
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
        # print(out_cls)
        # return out_src, out_cls

# import torch.utils.model_zoo as model_zoo
# model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))

class VGGDiscriminator(nn.Module):
    def __init__(self, image_size, cfg, curr_dim, c_dim, norm_layer, batch_norm):
        super(VGGDiscriminator, self).__init__()
        self.features, self.repeat_num= self.build_downsample(cfg, c_dim, norm_layer, batch_norm)
        self.features = nn.Sequential(*self.features)
        kernel_size = int(image_size / np.power(2, self.repeat_num))
        self.avgpool = nn.AvgPool2d(kernel_size, stride=1)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1)  # , bias=False
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=3, stride=1, padding=1)  # , bias=False

    def build_downsample(self, cfg, c_dim, norm_layer, batch_norm=False):
        model = []
        num = 0
        in_channels = 3
        for v in cfg:
            if v == 'M':
                model += [nn.MaxPool2d(kernel_size=2, stride=2)]
                num += 1
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=batch_norm)
                if batch_norm:
                    model += [conv2d, norm_layer(v), nn.ReLU(inplace=True)]
                else:
                    model += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return model, num

    def forward(self, x):
        h = self.features(x)
        out_pool = self.avgpool(h)
        out_src = self.conv1(out_pool)
        out_cls = self.conv2(out_pool) # 1*20*1*1
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


# model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
#                              kernel_size=3, stride=2,
#                              padding=1, output_padding=1,
#                              bias=use_bias),
class NetCGenerator(nn.Module):
    def __init__(self, output_nc, c_dim, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, block_channels=512, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(NetCGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # cfg =[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        model=[]
        # cfg =[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M']
        # cfg =[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M']
        # cfg =[64, 64, 'M', 128, 128, 'M', 256, 256, 256,'M', 512, 512, 512]
        # model += [DownsamplePool(cfg, c_dim, norm_layer=norm_layer, batch_norm=use_bias)]
        cfg =[64, 'M', 128, 'M', 256, 256,'M', 512, 512]
        model += [DownsampleStrided(cfg, c_dim, norm_layer=norm_layer, batch_norm=use_bias)]
        # model = make_downsample(cfg, c_dim, norm_layer=norm_layer, batch_norm=use_bias)
        for i in range(n_blocks):
            model += [DilatedBlock(block_channels, padding_type=padding_type, rates=2, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)]
        # dcfg = [256, 256, 256, 'M', 128, 128, 'M', 64, 64, 'M','W']
        # dcfg = [512, 512, 512, 'M',  256, 256, 256, 'M', 128, 128, 'M', 64, 64, 'M','W']        
        # dcfg = [512, 512, 512, 'M', 256, 256, 256, 'M', 128, 128, 'M', 64, 64, 'W']
        # model += [UpsamplePool(output_nc, dcfg, c_dim, block_channels, norm_layer=norm_layer, batch_norm=use_bias)]
        dcfg = [512, 512, 'M', 256, 256, 'M', 128, 'M', 64, 'W']
        model += [UpsampleStrided(output_nc, dcfg, c_dim, block_channels, norm_layer=norm_layer, batch_norm=use_bias)]
        # model += [Upsample(output_nc, dcfg, c_dim, block_channels=512, norm_layer=norm_layer, batch_norm=use_bias)]
        # model += make_upsample(dcfg, c_dim, block_channels=512, norm_layer=norm_layer, batch_norm=use_bias)
        self.model = nn.Sequential(*model)

    def forward(self, x, c):
        # x：1*3*321*321，c:1*20
        # Replicate spatially and concatenate domain information.
        # c = c.view(1, 20, 1, 1)
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))  # 1x20x321x321
        x = torch.cat([x, c], dim=1)  # 1x23x321x321
        return self.model(x)

# netG = NetCGenerator(output_nc, c_dim, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks, block_channels=block_channels, padding_type='reflect')
# image,label --> SegGenerator --> image,segmap
class SegGenerator(nn.Module):
    def __init__(self, output_nc, c_dim, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, block_channels=512, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(NetCGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # cfg =[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        model=[]
        # cfg =[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M']
        # cfg =[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M']
        
        # cfg =[64, 64, 'M', 128, 128, 'M', 256, 256, 256,'M', 512, 512, 512]
        # model += [DownsamplePool(cfg, c_dim, norm_layer=norm_layer, batch_norm=use_bias)]

        cfg =[64, 'M', 128, 'M', 256, 256,'M', 512, 512]
        model += [DownsampleStrided(cfg, c_dim, norm_layer=norm_layer, batch_norm=use_bias)]
        # model = make_downsample(cfg, c_dim, norm_layer=norm_layer, batch_norm=use_bias)
        for i in range(n_blocks):
            model += [DilatedBlock(block_channels, padding_type=padding_type, rates=2, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)]
        # dcfg = [512, 512, 512, 'M', 256, 256, 256, 'M', 128, 128, 'M', 64, 64, 'W']
        # model += [UpsamplePool(output_nc, dcfg, c_dim, block_channels, norm_layer=norm_layer, batch_norm=use_bias)]
        # dcfg = [256, 256, 256, 'M', 128, 128, 'M', 64, 64, 'M','W']
        # dcfg = [512, 512, 512, 'M',  256, 256, 256, 'M', 128, 128, 'M', 64, 64, 'M','W']   
        dcfg = [512, 512, 'M', 256, 256, 'M', 128, 'M', 64]     
        # dcfg = [512, 512, 'M', 256, 256, 'M', 128, 'M', 64, 'W']     
        model += [UpsampleStrided(output_nc, dcfg, c_dim, block_channels, norm_layer=norm_layer, batch_norm=use_bias)]
        # model += [nn.Conv2d(in_channels, output_nc, kernel_size=3, padding=1)]
        #         model += [nn.Tanh()]
        # model += [Upsample(output_nc, dcfg, c_dim, block_channels=512, norm_layer=norm_layer, batch_norm=use_bias)]
        # model += make_upsample(dcfg, c_dim, block_channels=512, norm_layer=norm_layer, batch_norm=use_bias)
        self.model = nn.Sequential(*model)
        self.conv1 = [nn.Conv2d(dcfg[-1], output_nc, kernel_size=3, padding=1), nn.Tanh()]
        # background class 21
        self.conv2 = [nn.Conv2d(dcfg[-1], c_dim+1, kernel_size=3, padding=1), nn.Tanh()]

    def forward(self, x, c):
        # x：1*3*321*321，c:1*20
        # Replicate spatially and concatenate domain information.
        # c = c.view(1, 20, 1, 1)
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))  # 1x20x321x321
        x = torch.cat([x, c], dim=1)  # 1x23x321x321
        out = self.model(x)
        image = self.conv1(out)
        segmap = self.conv2(out)
        return image, segmap

# netG = NetCGenerator(output_nc, 0, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks, block_channels=block_channels, padding_type='reflect')
class NetGenerator(nn.Module):
    def __init__(self, output_nc, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, block_channels=512, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(NetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # cfg =[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        model=[]
        # cfg =[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M']
        # cfg =[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M']
        
        # cfg =[64, 64, 'M', 128, 128, 'M', 256, 256, 256,'M', 512, 512, 512]
        # model += [DownsamplePool(cfg, 0, norm_layer=norm_layer, batch_norm=use_bias)]

        cfg =[64, 'M', 128, 'M', 256, 256,'M', 512, 512]
        model += [DownsampleStrided(cfg, 0, norm_layer=norm_layer, batch_norm=use_bias)]
        # model = make_downsample(cfg, c_dim, norm_layer=norm_layer, batch_norm=use_bias)
        for i in range(n_blocks):
            model += [DilatedBlock(block_channels, padding_type=padding_type, rates=2, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)]
        dcfg = [512, 512, 512, 'M', 256, 256, 256, 'M', 128, 128, 'M', 64, 64, 'W']
        # dcfg = [256, 256, 256, 'M', 128, 128, 'M', 64, 64, 'M','W']
        # dcfg = [512, 512, 512, 'M',  256, 256, 256, 'M', 128, 128, 'M', 64, 64, 'M','W']        
        model += [UpsamplePool(output_nc, dcfg, 0, block_channels, norm_layer=norm_layer, batch_norm=use_bias)]
        # model += [Upsample(output_nc, dcfg, c_dim, block_channels=512, norm_layer=norm_layer, batch_norm=use_bias)]
        # model += make_upsample(dcfg, c_dim, block_channels=512, norm_layer=norm_layer, batch_norm=use_bias)
        self.model = nn.Sequential(*model)

    def forward(self, x):
        # x：1*3*321*321，c:1*20
        # Replicate spatially and concatenate domain information.
        # c = c.view(1, 20, 1, 1)
        # c = c.view(c.size(0), c.size(1), 1, 1)
        # c = c.repeat(1, 1, x.size(2), x.size(3))  # 1x20x321x321
        # x = torch.cat([x, c], dim=1)  # 1x23x321x321
        out = self.model(x)
        return out

class DownsamplePool(nn.Module):
    def __init__(self, cfg, c_dim, norm_layer, batch_norm=False):
        super(DownsamplePool, self).__init__()
        self.DownsamplePool = self.build_downsample(cfg, c_dim, norm_layer, batch_norm=False)

    def build_downsample(self, cfg, c_dim, norm_layer, batch_norm=False):
        model = []
        in_channels = 3 + c_dim
        for v in cfg:
            if v == 'M':
                model += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=batch_norm)
                # if batch_norm:
                model += [conv2d, norm_layer(v), nn.ReLU(inplace=True)]
                # else:
                #     model += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*model)

    def forward(self, x):
        out = self.DownsamplePool(x)
        return out

class DownsampleStrided(nn.Module):
    def __init__(self, cfg, c_dim, norm_layer, batch_norm=False):
        super(DownsampleStrided, self).__init__()
        self.DownsampleStrided = self.build_downsample(cfg, c_dim, norm_layer, batch_norm=False)

    def build_downsample(self, cfg, c_dim, norm_layer, batch_norm=False):
        model = []
        in_channels = 3 + c_dim
        for v in cfg:
            if v == 'M':
                model += [nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=batch_norm),norm_layer(in_channels), nn.ReLU(inplace=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=batch_norm)
                # if batch_norm:
                model += [conv2d, norm_layer(v), nn.ReLU(inplace=True)]
                # else:
                    # model += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*model)

    def forward(self, x):
        out = self.DownsampleStrided(x)
        return out

# model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
#                     stride=2, padding=1, bias=use_bias),
#           norm_layer(ngf * mult * 2),
#           nn.ReLU(True)]

class UpsamplePool(nn.Module):
    def __init__(self, output_nc, dcfg, c_dim, block_channels, norm_layer, batch_norm=False):
        super(Upsample, self).__init__()
        self.Upsample = self.build_upsample(output_nc, dcfg, c_dim, block_channels, norm_layer, batch_norm)

    def build_upsample(self, output_nc, dcfg, c_dim, block_channels, norm_layer, batch_norm):
        model = []
        in_channels = block_channels
        # dcfg = [512, 512, 512, 'M',  256, 256, 256, 'M', 128, 128, 'M', 64, 64, 'M', 'W']
        for v in dcfg:
            if v == 'M':
                dconv2d = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1, groups=1, bias=batch_norm)
                model += [dconv2d, norm_layer(in_channels), nn.ReLU(inplace=True)]
            elif v == 'W':
                # model += [nn.ReflectionPad2d(3)]
                model += [nn.Conv2d(in_channels, output_nc, kernel_size=3, padding=1)]
                model += [nn.Tanh()]
            else:
                dconv2d = nn.ConvTranspose2d(in_channels, v, kernel_size=3, stride=1, padding=1, output_padding=0, groups=1, bias=batch_norm)
                if batch_norm:
                    model += [dconv2d, norm_layer(v), nn.ReLU(inplace=True)]
                else:
                    model += [dconv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*model)

    def forward(self, x):
        out = self.Upsample(x)
        return out

class UpsampleStrided(nn.Module):
    def __init__(self, output_nc, dcfg, c_dim, block_channels, norm_layer, batch_norm=False):
        super(UpsampleStrided, self).__init__()
        self.Upsample = self.build_upsample(output_nc, dcfg, c_dim, block_channels, norm_layer, batch_norm)

    def build_upsample(self, output_nc, dcfg, c_dim, block_channels, norm_layer, batch_norm):
        model = []
        in_channels = block_channels
        # dcfg = [512, 512, 512, 'M',  256, 256, 256, 'M', 128, 128, 'M', 64, 64, 'M', 'W']
        for v in dcfg:
            if v == 'M':
                dconv2d = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1, groups=1, bias=batch_norm)
                model += [dconv2d, norm_layer(in_channels), nn.ReLU(inplace=True)]
            elif v == 'W':
                # model += [nn.ReflectionPad2d(3)]
                model += [nn.Conv2d(in_channels, output_nc, kernel_size=3, padding=1)]
                model += [nn.Tanh()]
            else:
                dconv2d = nn.ConvTranspose2d(in_channels, v, kernel_size=3, stride=1, padding=1, output_padding=0, groups=1, bias=batch_norm)
                if batch_norm:
                    model += [dconv2d, norm_layer(v), nn.ReLU(inplace=True)]
                else:
                    model += [dconv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*model)

    def forward(self, x):
        out = self.Upsample(x)
        return out


# Define a resnet block
class DilatedBlock(nn.Module):
    def __init__(self, dim, padding_type, rates, norm_layer, use_dropout, use_bias):
        super(DilatedBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, rates, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, rates, norm_layer, use_dropout, use_bias):
        conv_block = []
        # p = 0
        # if padding_type == 'reflect':
        #     conv_block += [nn.ReflectionPad2d(1)]
        # elif padding_type == 'replicate':
        #     conv_block += [nn.ReplicationPad2d(1)]
        # elif padding_type == 'zero':
        #     p = 1
        # else:
        #     raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        # nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, dilation=1, bias=use_bias)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=2, dilation=rates, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.conv_block(x)
        return out
