'''
File Created: Thursday, 18th April 2019 2:24:37 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''

import sys
sys.path.append("../")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, InstanceNorm2d, Conv3d, InstanceNorm3d, Conv2d, InstanceNorm2d, ConvTranspose2d
from det3.methods.voxelnet.layer import VFELayer

class FeatureNet(nn.Module):
    '''
    FeatureNet of VoxelNet
    '''
    def __init__(self, in_channels=7, out_gridsize=(5, 8, 10)):
        '''
        inputs:
            in_channels: (int)
                the # of feature channel of input
            out_gridsize: (tuple) (int, int, int)
                the size of output grid
        '''
        super(FeatureNet, self).__init__()
        self.vfe1 = VFELayer(in_channels=in_channels, out_channels=32)
        self.vfe2 = VFELayer(in_channels=32, out_channels=128)
        self.dense = Linear(128, 128, bias=True)
        self.instance_norm = InstanceNorm2d(128)
        self.out_gridsize = out_gridsize

    def forward(self, x, coordinate):
        '''
        inputs:
            x (Tensor) [#batch, #vox, #pts, in_channels]
                input voxelized point clouds
            coordinate (Tensor) [#batch, $vox, 3]:
                coordinate has to be compatible with the self.out_gridsize
        outputs:
            grid (Tensor) [#batch, out_channels, H(z), W(y), L(x)] (z, y, x is in LiDAR frame)
        Note:
            The input x is in shape [#batch, #vox, #pts, in_channels],
            but the #vox is different from batches.
            It can only be trained with one batch for now.
        '''
        # x [#batch, #vox, #pts, in_channels]
        num_batch = x.shape[0]
        mask = torch.sum(x, dim=-1)
        mask = torch.eq(mask, 0.0).unsqueeze(-1).type(torch.float32)
        assert mask.sum() != x.shape[0] * x.shape[1] * x.shape[2], "ERROR!, Mask should not be all zeros!"
        # x [#batch, #vox, #pts, in_channels]
        x = self.vfe1(x, mask)
        # x [#batch, #vox, #pts, 32]
        x = self.vfe2(x, mask)
        # x [#batch, #vox, #pts, 128]
        x = self.dense(x)
        # x [#batch, #vox, #pts, 128]
        x = x.permute(0, 3, 1, 2)
        # x [#batch, 128, #vox, #pts]
        x = self.instance_norm(x)
        # x [#batch, 128, #vox, #pts]
        x = F.relu(x)
        # x [#batch, 128, #vox, #pts]
        x, _ = torch.max(x, dim=3, keepdim=False)
        # x [#batch, 128, #vox]
        grid = torch.zeros(num_batch, 128, self.out_gridsize[0], self.out_gridsize[1], self.out_gridsize[2]).cuda()
        for i in range(num_batch):
            grid[i, :, coordinate[i, :, 0], coordinate[i, :, 1], coordinate[i, :, 2]] += x[i, ::]
        return grid

class MiddleLayer(nn.Module):
    '''
    MiddleLayer of VoxelNet
    '''
    def __init__(self):
        super(MiddleLayer, self).__init__()
        self.layer1 = Conv3d(128, 64, [3, 3, 3], [2, 1, 1], padding=[1, 1, 1], dilation=1, groups=1, bias=True)
        self.layer1_inorm = InstanceNorm3d(64)
        self.layer2 = Conv3d(64, 64, [3, 3, 3], [1, 1, 1], padding=[0, 1, 1], dilation=1, groups=1, bias=True)
        self.layer2_inorm = InstanceNorm3d(64)
        self.layer3 = Conv3d(64, 64, [3, 3, 3], [2, 1, 1], padding=[1, 1, 1], dilation=1, groups=1, bias=True)
        self.layer3_inorm = InstanceNorm3d(64)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer1_inorm(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = self.layer2_inorm(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = self.layer3_inorm(x)
        x = F.relu(x)
        x = torch.cat([x[:, :, 0, :, :], x[:, :, 1, :, :]], dim=1)
        return x

class RPN(nn.Module):
    '''
    RPN of VoxelNet
    '''
    def __init__(self):
        super(RPN, self).__init__()
        # block1
        self.block1_conv1 = Conv2d(128, 128, [3, 3], [2, 2], padding=[1, 1], dilation=1, groups=1, bias=True)
        self.block1_inorm1 = InstanceNorm2d(128)
        self.block1_conv2 = Conv2d(128, 128, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=True)
        self.block1_inorm2 = InstanceNorm2d(128)
        self.block1_conv3 = Conv2d(128, 128, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=True)
        self.block1_inorm3 = InstanceNorm2d(128)
        self.block1_conv4 = Conv2d(128, 128, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=True)
        self.block1_inorm4 = InstanceNorm2d(128)
        self.block1_trconv5 = ConvTranspose2d(128, 256, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=True)
        self.block1_inorm5 = InstanceNorm2d(256)
        # block2
        self.block2_conv1 = Conv2d(128, 128, [3, 3], [2, 2], padding=[1, 1], dilation=1, groups=1, bias=True)
        self.block2_inorm1 = InstanceNorm2d(128)
        self.block2_conv2 = Conv2d(128, 128, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=True)
        self.block2_inorm2 = InstanceNorm2d(128)
        self.block2_conv3 = Conv2d(128, 128, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=True)
        self.block2_inorm3 = InstanceNorm2d(128)
        self.block2_conv4 = Conv2d(128, 128, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=True)
        self.block2_inorm4 = InstanceNorm2d(128)
        self.block2_conv5 = Conv2d(128, 128, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=True)
        self.block2_inorm5 = InstanceNorm2d(128)
        self.block2_conv6 = Conv2d(128, 128, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=True)
        self.block2_inorm6 = InstanceNorm2d(128)
        self.block2_trconv7 = ConvTranspose2d(128, 256, [2, 2], [2, 2], padding=[0,0], dilation=1, groups=1, bias=True)
        self.block2_inorm7 = InstanceNorm2d(256)
        # block3
        self.block3_conv1 = Conv2d(128, 256, [3, 3], [2, 2], padding=[1, 1], dilation=1, groups=1, bias=True)
        self.block3_inorm1 = InstanceNorm2d(256)
        self.block3_conv2 = Conv2d(256, 256, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=True)
        self.block3_inorm2 = InstanceNorm2d(256)
        self.block3_conv3 = Conv2d(256, 256, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=True)
        self.block3_inorm3 = InstanceNorm2d(256)
        self.block3_conv4 = Conv2d(256, 256, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=True)
        self.block3_inorm4 = InstanceNorm2d(256)
        self.block3_conv5 = Conv2d(256, 256, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=True)
        self.block3_inorm5 = InstanceNorm2d(256)
        self.block3_conv6 = Conv2d(256, 256, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=True)
        self.block3_inorm6 = InstanceNorm2d(256)
        self.block3_trconv7 = ConvTranspose2d(256, 256, [4, 4], [4, 4], padding=[0,0], dilation=1, groups=1, bias=True)
        self.block3_inorm7 = InstanceNorm2d(256)

        # head
        self.head_conv_cls = Conv2d(768, 2, [1, 1], [1, 1], padding=[0, 0], dilation=1, groups=1, bias=True)
        self.head_conv_reg = Conv2d(768, 14, [1, 1], [1, 1], padding=[0, 0], dilation=1, groups=1, bias=True)
    def forward(self, x):
        # block1
        x = self.block1_conv1(x)
        x = self.block1_inorm1(x)
        x = F.relu(x)
        x = self.block1_conv2(x)
        x = self.block1_inorm2(x)
        x = F.relu(x)
        x = self.block1_conv3(x)
        x = self.block1_inorm3(x)
        x = F.relu(x)
        x = self.block1_conv4(x)
        x = self.block1_inorm4(x)
        x = F.relu(x)
        x_1 = self.block1_trconv5(x)
        x_1 = self.block1_inorm5(x_1)
        x_1 = F.relu(x_1)

        # block2
        x = self.block2_conv1(x)
        x = self.block2_inorm1(x)
        x = F.relu(x)
        x = self.block2_conv2(x)
        x = self.block2_inorm2(x)
        x = F.relu(x)
        x = self.block2_conv3(x)
        x = self.block2_inorm3(x)
        x = F.relu(x)
        x = self.block2_conv4(x)
        x = self.block2_inorm4(x)
        x = F.relu(x)
        x = self.block2_conv5(x)
        x = self.block2_inorm5(x)
        x = F.relu(x)
        x = self.block2_conv6(x)
        x = self.block2_inorm6(x)
        x = F.relu(x)
        x_2 = self.block2_trconv7(x)
        x_2 = self.block2_inorm7(x_2)
        x_2 = F.relu(x_2)

        # block3
        x = self.block3_conv1(x)
        x = self.block3_inorm1(x)
        x = F.relu(x)
        x = self.block3_conv2(x)
        x = self.block3_inorm2(x)
        x = F.relu(x)
        x = self.block3_conv3(x)
        x = self.block3_inorm3(x)
        x = F.relu(x)
        x = self.block3_conv4(x)
        x = self.block3_inorm4(x)
        x = F.relu(x)
        x = self.block3_conv5(x)
        x = self.block3_inorm5(x)
        x = F.relu(x)
        x = self.block3_conv6(x)
        x = self.block3_inorm6(x)
        x = F.relu(x)
        x_3 = self.block3_trconv7(x)
        x_3 = self.block3_inorm7(x_3)
        x_3 = F.relu(x_3)

        x_ = torch.cat([x_1, x_2, x_3], 1)
        pmap = self.head_conv_cls(x_)
        pmap = torch.sigmoid(pmap)
        rmap = self.head_conv_reg(x_)
        return pmap, rmap


if __name__ == "__main__":
    data = torch.randn(1, 3, 35, 7) # [#batch, #vox, #pts, #feature]
    coordinate = torch.zeros(1, 3, 3).long() # [#batch, #vox, 3(z, y, x)]
    out_gridsize = (10, 400, 352) #[H(z), W(y), L(x)]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            coordinate[i, j, :] = torch.LongTensor([torch.randint(out_gridsize[0], size=(1,)),
                                                    torch.randint(out_gridsize[1], size=(1,)),
                                                    torch.randint(out_gridsize[2], size=(1,))])
    featurenet = FeatureNet(in_channels=7, out_gridsize=out_gridsize).cuda()
    middlelayer = MiddleLayer().cuda()
    rpn = RPN().cuda()
    result = featurenet(data.cuda(), coordinate.cuda())
    result = middlelayer(result)
    result = rpn(result)
    print("DONE")