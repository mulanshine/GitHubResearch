import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# 'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

'''
Modified from https://github.com/pytorch/vision.git
'''
import math
import torch.nn as nn
import torch.nn.init as init
import torch
# import torch.optim as optim
# from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
import time
import os
# import copy


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
class VGGDiscriminator(nn.Module):
    def __init__(self, image_size=256, cfg,curr_dim=512, c_dim=20, norm_layer, batch_norm=False):
        super(VGGDiscriminator, self).__init__()
        self.Downsample, self.repeat_num= self.build_downsample(cfg, c_dim, norm_layer, batch_norm=False)
        kernel_size = int(image_size / np.power(2, self.repeat_num))
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size, stride=1)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=3, stride=1, padding=1, bias=False)

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
        return nn.Sequential(*model), num

    def forward(self, x):
        h = self.Downsample(x)
        out_src = self.conv1(h)
        out_pool = self.avgpool(h)
        out_cls = self.conv2(out_pool) # 1*20*1*1
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1)) 


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']











##########################################################################################
        还没写初始化load from pre_model
##########################################################################################

    # if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))