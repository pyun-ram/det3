'''
File Created: Thursday, 18th April 2019 2:24:13 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import torch
import torch.nn as nn
from torch.nn import Linear, BatchNorm2d
import torch.nn.functional as F

class VFELayer(nn.Module):
    '''
    Voxel Feature Encoding Layer
    '''
    def __init__(self, in_channels, out_channels):
        '''
        inputs:
            in_channels: (int)
            out_channels: (int)
        '''
        super(VFELayer, self).__init__()
        self.units = int(out_channels /2)
        self.dense = Linear(in_channels, self.units, bias=True)
        self.bn = BatchNorm2d(self.units) # TODO: Batch Norm or Instance Norm?

    def forward(self, x, mask):
        '''
        inputs:
            x: (torch.tensor) [#batch, #vox, #pts, in_channels]
            mask: (torch.tensor) [#batch, #vox, #pts, 1]
        return:
            troch.tensor [#batch, #vox, #pts, out_channels]
        '''
        # x in shape [#batch, #vox, #pts, in_channels]
        num_pts = x.shape[2]
        x = self.dense(x)
        x = x.permute(0, 3, 2, 1)
        x = self.bn(x)
        x = F.relu(x)
        x = x.permute(0, 3, 2, 1)
        # global_feature in shape [#batch, #vox, 1, units]
        global_feature, _ = torch.max(x, dim=2, keepdim=True)
        concatenated = torch.cat([x, global_feature.repeat(1, 1, num_pts, 1)], dim=3)
        mask_ = mask.repeat(1, 1, 1, 2 * self.units)
        return mask_ * concatenated

if __name__ == "__main__":
    data = torch.randn(1, 500, 35, 7)
    mask = torch.randn(1, 500, 35, 1)
    vfelayer = VFELayer(in_channels=7, out_channels=128)
    result = vfelayer(data, mask)
    print(result.shape)

