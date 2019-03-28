'''
Modified from https://github.com/pytorch/vision.git
'''
import math
import torch.nn as nn
import torch.nn.init as init

__all__ = ['VGG','vgg16', 'vgg16_bn']

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

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M']  #conv_5: 512, 512, 512, 'M'
dcfg = [512, 512, 512, 'M',  256, 256, 256, 'M', 128, 128, 'M', 64, 64, 'M'] 

class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self):
        super(VGG, self).__init__()
        self.model = make_features(cfg, c)
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(True),
        #     nn.Linear(512, 10),
        # )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x, c):
        x = self.features(x, c)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# def make_features(cfg,dcfg, c, batch_norm=False)
#     layers = []
#     n_input = 3 + len(c)
#     in_channels = n_input
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     layers = make_blocks(layers,num_blocks=n_blocks, dim=512, batch_norm=False)
#     for dv in dcfg:
###########################这里有问题，'M'这里upsample怎样增大size########################################################
#         if dv == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             dconv2d = nn.ConvTranspose2d(in_channels, dv, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [dconv2d, nn.BatchNorm2d(dv), nn.ReLU(inplace=True)]
#             else:
#                 layers += [dconv2d, nn.ReLU(inplace=True)]
#             in_channels = dv
#     layers += nn.ConvTranspose2d(in_channels,n_input,kernel_size=3, padding=1)
#     return nn.Sequential(*layers)

# def make_layers(cfg, c, batch_norm=False):
#     # norm_layer = get_norm_layer(norm_type=norm)
#     layers = []
#     in_channels = 3 + len(c)
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     return layers

# def make_blocks(layers,num_blocks=n_blocks, dim=512, batch_norm=False):
#     # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
#     for n in num_blocks:
#         layers += [nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, dilation=1, bias=use_bias)]
#         if batch_norm:
#             layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#         else:
#             layers += [conv2d, nn.ReLU(inplace=True)]
#         layers += [nn.ReLU(True)]
#     return layers

# def make_upsample(layers,dcfg)
#     for v in dcfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             dconv2d = nn.ConvTranspose2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [dconv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [dconv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     return layers
    
# def make_model(cfg, dcfg, c, num_blocks, dim=512, batch_norm=False):
    layers = make_layers(cfg, c, batch_norm=False)
    layers = make_blocks(layers,num_blocks, dim=512, batch_norm=False)
    layers = make_upsample(layers,dcfg)
    return nn.Sequential(*layers)

# class torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True)
# 2维的转置卷积操作（transposed convolution operator，注意改视作操作可视作解卷积操作，但并不是真正的解卷积操作） 该模块可以看作是Conv2d相对于其输入的梯度，有时（但不正确地）被称为解卷积操作。

# def vgg16():
#     """VGG 16-layer model (configuration "D")"""
#     return VGG(make_layers(cfg, c))


# def vgg16_bn():
#     """VGG 16-layer model (configuration "D") with batch normalization"""
#     return VGG(make_layers(cfg, c, batch_norm=True))
