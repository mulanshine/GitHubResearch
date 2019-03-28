import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
import torchvision.models as models



class PerceptualLoss(nn.Module):
    def __init__(self, perceptual_layers): #, gpu_ids
        super(PerceptualLoss, self).__init__()

        # self.gpu_ids = gpu_ids

        vgg = models.vgg16(pretrained=True).features
        self.vgg_submodel = nn.Sequential()
        for i,layer in enumerate(list(vgg)):
            self.vgg_submodel.add_module(str(i),layer)
            if i == perceptual_layers:
                break
        self.vgg_submodel = self.vgg_submodel.cuda() # gpu_ids
        # print('####perceptual sub model defination!####')
        # print(self.vgg_submodel)

    def forward(self, inputs, targets):
        # use conv1_2 of vgg19. first have to normalize inputs
        '''
            All pre-trained models expect input images normalized in the same way, 
        i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H 
        and W are expected to be at least 224. The images have to be loaded in to a
        range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and 
        std = [0.229, 0.224, 0.225].
        '''
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        fake_p2_norm = (inputs + 1)/2 # [-1, 1] => [0, 1]
        input_p2_norm = (targets + 1)/2 # [-1, 1] => [0, 1]
        fake_p2_norm = self.vgg_submodel(fake_p2_norm)
        input_p2_norm = self.vgg_submodel(input_p2_norm)
        input_p2_norm_no_grad = input_p2_norm.detach()
        loss_perceptual = F.l1_loss(fake_p2_norm, input_p2_norm_no_grad)
        return loss_perceptual
