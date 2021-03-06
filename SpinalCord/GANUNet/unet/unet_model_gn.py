# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch.nn as nn
from .unet_parts_gn import *

class UNet_GN7(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_GN7, self).__init__()
        self.inc = inconv(n_channels, 64, 32)
        self.down1 = down(64, 128, 32)
        self.down2 = down(128, 256, 32)
        self.down3 = down(256, 512, 32)
        self.down4 = down(512, 512, 32)
        self.down5 = down(512, 1024, 32)
        self.down6 = down(1024, 1024, 32)
        self.down7 = down(1024, 1024, 32)
        self.drop = nn.Dropout(p=0.5)
        self.up1 = up(2048, 1024, 32)
        self.up2 = up(2048, 512, 32)
        self.up3 = up(1024, 512, 32)
        self.up4 = up(1024, 256, 32)
        self.up5 = up(512, 128, 32)
        self.up6 = up(256, 64, 32)
        self.up7 = up(128, 64, 32)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        x8 = self.drop(x8)
        x = self.up1(x8, x7)
        x = self.up2(x, x6)
        x = self.up3(x, x5)
        x = self.up4(x, x4)
        x = self.up5(x, x3)
        x = self.up6(x, x2)
        x = self.up7(x, x1)
        x = self.outc(x)
        return F.sigmoid(x)
        # return F.softmax(x)


class UNet_GN5(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_GN5, self).__init__()
        self.inc = inconv(n_channels, 64, 32)
        self.down1 = down(64, 128, 32)
        self.down2 = down(128, 256, 32)
        self.down3 = down(256, 512, 32)
        self.down4 = down(512, 1024, 32)
        self.down5 = down(1024, 1024, 32)
        self.drop = nn.Dropout(p=0.5)
        self.up1 = up(2048, 512, 32)
        self.up2 = up(1024, 256, 32)
        self.up3 = up(512, 128, 32)
        self.up4 = up(256, 64, 32)
        self.up5 = up(128, 64, 32)
        self.outc = outconv(64, n_classes)
        

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x6 = self.drop(x6)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        x = self.outc(x)
        # x = F.softmax(x)
        # x = F.sigmoid(x)
        # return x
        return x
        # return 