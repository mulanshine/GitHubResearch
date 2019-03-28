# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


class tranconv1(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch,num_groups=32):
        super(tranconv1, self).__init__()
        out_ch = int(in_ch/2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.Conv2d(in_ch, out_ch, 1))

    def forward(self, x): 
        x = self.conv(x)
        return x

# class tranconv1(nn.Module):
#     '''(conv => BN => ReLU) * 2'''
#     def __init__(self, in_ch,num_groups=32):
#         super(tranconv1, self).__init__()
#         out_ch = int(in_ch/2)
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 1),
#             nn.GroupNorm(num_groups,out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 1),
#             nn.GroupNorm(num_groups,out_ch),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x): 
#         x = self.conv(x)
#         return x


# groupnorm
class double_conv_GN(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,num_groups=32):
        super(double_conv_GN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups,out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups,out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): 
        x = self.conv(x)
        return x


class double_inconv_GN(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,num_groups=32):
        super(double_inconv_GN, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(3),# 7,0
            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0),
            nn.GroupNorm(num_groups,out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups,out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch,num_groups=32):
        super(inconv, self).__init__()
        self.conv = double_inconv_GN(in_ch, out_ch, num_groups)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=32):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv_GN(in_ch, out_ch,num_groups)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=32, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv_GN(in_ch, out_ch,num_groups)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

