'''
File Created: Wednesday, 27th March 2019 8:43:24 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import torch
import torch.nn as nn
from torch.nn import Conv3d, BatchNorm3d
import torch.nn.functional as F

class FCN3D(nn.Module):
    '''
    3D FCN Model
    '''
    def __init__(self):
        super(FCN3D, self).__init__()
        # Input shape: [N, C, D, H, W]
        self.layer1 = Conv3d(1, 32, [5, 5, 5], [2, 2, 2], padding=2, dilation=1, groups=1, bias=False)
        self.layer1_bn = BatchNorm3d(32)
        self.layer2 = Conv3d(32, 64, [5, 5, 5], [2, 2, 2], padding=2, dilation=1, groups=1, bias=False)
        self.layer2_bn = BatchNorm3d(64)
        self.layer3 = Conv3d(64, 96, [3, 3, 3], [1, 1, 1], padding=1, dilation=1, groups=1, bias=False)
        self.layer3_bn = BatchNorm3d(96)
        self.layer4 = Conv3d(96, 96, [3, 3, 3], [1, 1, 1], padding=1, dilation=1, groups=1, bias=False)
        self.layer4_bn = BatchNorm3d(96)
        self.layer_obj = Conv3d(96, 1, [3, 3, 3], [1, 1, 1], padding=1, dilation=1, groups=1, bias=False)
        self.layer_reg = Conv3d(96, 24, [3, 3, 3], [1, 1, 1], padding=1, dilation=1, groups=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer1_bn(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer2_bn(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.layer3_bn(x)
        x = self.layer4(x)
        x = F.relu(x)
        x = self.layer4_bn(x)
        obj = self.layer_obj(x)
        obj = torch.sigmoid(obj)
        reg = self.layer_reg(x)

        return obj, reg

if __name__ == '__main__':
    x = torch.randn(2, 1, 40, 800, 800).cuda(0)
    model = FCN3D().cuda(0)
    obj, reg = model(x)
    print(x.shape)
    print(obj.shape)
    print(reg.shape)
    input()
