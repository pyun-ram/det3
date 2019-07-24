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
from torch.nn import Linear, Conv3d, Conv2d, ConvTranspose2d, BatchNorm2d, BatchNorm3d
import spconv
from det3.methods.voxelnet.layer import VFELayer
from dropblock import DropBlock2D

# It is from https://github.com/traveller59/torchplus
import inspect
def get_pos_to_kw_map(func):
    pos_to_kw = {}
    fsig = inspect.signature(func)
    pos = 0
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            pos_to_kw[pos] = name
        pos += 1
    return pos_to_kw
# It is from https://github.com/traveller59/torchplus
def change_default_args(**kwargs):
    def layer_wrapper(layer_class):
        class DefaultArgLayer(layer_class):
            def __init__(self, *args, **kw):
                pos_to_kw = get_pos_to_kw_map(layer_class.__init__)
                kw_to_pos = {kw: pos for pos, kw in pos_to_kw.items()}
                for key, val in kwargs.items():
                    if key not in kw and kw_to_pos[key] > len(args):
                        kw[key] = val
                super().__init__(*args, **kw)

        return DefaultArgLayer

    return layer_wrapper
# It is from https://github.com/traveller59/torchplus
class Empty(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Empty, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        return args

class FeatureNet(nn.Module):
    '''
    FeatureNet of VoxelNet
    '''
    def __init__(self, in_channels=7, out_gridsize=(5, 8, 10), bool_sparse=False):
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
        self.norm = BatchNorm2d(128)
        self.out_gridsize = out_gridsize
        self.bool_sparse = bool_sparse
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # TODO: Note InstanceNorm2d is not initialized

    def forward(self, x, coordinate, batch_size):
        '''
        inputs:
            x (Tensor) [#vox, #pts, in_channels]
                input voxelized point clouds
            coordinate (Tensor) [$vox, 1+3]:
                coordinate has to be compatible with the self.out_gridsize
        outputs:
            grid (Tensor) [#batch, out_channels, H(z), W(y), L(x)] (z, y, x is in LiDAR frame)
        Note:
            The input x is in shape [#batch, #vox, #pts, in_channels],
            but the #vox is different from batches.
            It can only be trained with one batch for now.
        '''
        # x [#batch, #vox, #pts, in_channels]
        num_batch = torch.max(coordinate[:, 0]) + 1
        assert num_batch == batch_size
        x = x.unsqueeze(0)
        mask = torch.sum(x, dim=-1)
        mask = torch.eq(mask, 0.0).unsqueeze(-1).type(torch.float32) # empty point position is one, occupied position is zero
        mask = 1 - mask # empty point position is zero, occupied position is one
        assert mask.sum() > 0, "ERROR!, Mask should not be all zeros!"
        # x [#batch, #vox, #pts, in_channels]
        x = self.vfe1(x, mask)
        # x [#batch, #vox, #pts, 32]
        x = self.vfe2(x, mask)
        # x [#batch, #vox, #pts, 128]
        x = self.dense(x)
        # x [#batch, #vox, #pts, 128]
        x = x.permute(0, 3, 1, 2)
        # x [#batch, 128, #vox, #pts]
        x = self.norm(x)
        # x [#batch, 128, #vox, #pts]
        x = F.leaky_relu(x, negative_slope=0.1)
        # x [#batch, 128, #vox, #pts]
        x, _ = torch.max(x, dim=3, keepdim=False)
        # x [#batch, 128, #vox]
        if not self.bool_sparse:
            # grid = torch.zeros(num_batch, 128, self.out_gridsize[0],
            #                    self.out_gridsize[1], self.out_gridsize[2]).cuda()
            # for i in range(num_batch):
            #     grid[i, :, coordinate[i, :, 0], coordinate[i, :, 1], coordinate[i, :, 2]] += x[i, ::]
            # return grid
            print("Error: The dense version of VoxelNet is discarded.")
            raise NotImplementedError
        else:
            return x, coordinate

class MiddleLayer(nn.Module):
    '''
    MiddleLayer of VoxelNet
    '''
    def __init__(self):
        super(MiddleLayer, self).__init__()
        self.layer1 = Conv3d(128, 64, [3, 3, 3], [2, 1, 1], padding=[1, 1, 1],
                             dilation=1, groups=1, bias=True)
        self.layer1_norm = BatchNorm3d(64)
        self.layer2 = Conv3d(64, 64, [3, 3, 3], [1, 1, 1], padding=[0, 1, 1],
                             dilation=1, groups=1, bias=True)
        self.layer2_norm = BatchNorm3d(64)
        self.layer3 = Conv3d(64, 64, [3, 3, 3], [2, 1, 1], padding=[1, 1, 1],
                             dilation=1, groups=1, bias=True)
        self.layer3_norm = BatchNorm3d(64)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # TODO: Note InstanceNorm3d is not initialized

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer1_norm(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.layer2(x)
        x = self.layer2_norm(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.layer3(x)
        x = self.layer3_norm(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = torch.cat([x[:, :, 0, :, :], x[:, :, 1, :, :]], dim=1)
        return x

class SparseMiddleLayer(nn.Module):
    '''
    MiddleLayer of VoxelNet (SPConv imporvement)
    '''
    def __init__(self, sparse_shape):
        super(SparseMiddleLayer, self).__init__()
        self.sparse_shape = sparse_shape
        BatchNorm1d = change_default_args(
            eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
        # BatchNorm1d = Empty
        SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
        SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(128, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            nn.LeakyReLU(negative_slope=0.1),
            SubMConv3d(16, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            nn.LeakyReLU(negative_slope=0.1),
            SpConv3d(16, 32, 3, 2,
                     padding=1),  # [1600, 1200, 41] -> [800, 600, 21]
            BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.1),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.1),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.1),
            SpConv3d(32, 64, 3, 2,
                     padding=1),  # [800, 600, 21] -> [400, 300, 11]
            BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.1),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.1),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.1),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.1),
            SpConv3d(64, 64, 3, 2,
                     padding=[0, 1, 1]),  # [400, 300, 11] -> [200, 150, 5]
            BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.1),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.1),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.1),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.1),
            SpConv3d(64, 64, (3, 1, 1),
                     (2, 1, 1)),  # [200, 150, 5] -> [200, 150, 2]
            BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.1),
        )

    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        # t = time.time()
        # torch.cuda.synchronize()
        ret = self.middle_conv(ret)
        # torch.cuda.synchronize()
        # print("spconv forward time", time.time() - t)
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret

class RPN(nn.Module):
    '''
    RPN of VoxelNet
    '''
    def __init__(self, bool_sparse=False):
        super(RPN, self).__init__()
        # block1
        self.bool_sparse = bool_sparse
        if not self.bool_sparse:
            self.block1_conv1 = Conv2d(128, 128, [3, 3], [2, 2],
                                       padding=[1, 1], dilation=1, groups=1, bias=False)
        else:
            self.block1_conv1 = Conv2d(128, 128, [3, 3], [1, 1],
                                       padding=[1, 1], dilation=1, groups=1, bias=False)
        self.block1_norm1 = BatchNorm2d(128)
        self.block1_conv2 = Conv2d(128, 128, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=False)
        self.block1_norm2 = BatchNorm2d(128)
        self.block1_conv3 = Conv2d(128, 128, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=False)
        self.block1_norm3 = BatchNorm2d(128)
        self.block1_conv4 = Conv2d(128, 128, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=False)
        self.block1_norm4 = BatchNorm2d(128)
        self.block1_trconv5 = ConvTranspose2d(128, 256, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=False)
        self.block1_norm5 = BatchNorm2d(256)
        # block2
        self.block2_conv1 = Conv2d(128, 128, [3, 3], [2, 2], padding=[1, 1], dilation=1, groups=1, bias=False)
        self.block2_norm1 = BatchNorm2d(128)
        self.block2_conv2 = Conv2d(128, 128, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=False)
        self.block2_norm2 = BatchNorm2d(128)
        self.block2_conv3 = Conv2d(128, 128, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=False)
        self.block2_norm3 = BatchNorm2d(128)
        self.block2_conv4 = Conv2d(128, 128, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=False)
        self.block2_norm4 = BatchNorm2d(128)
        self.block2_conv5 = Conv2d(128, 128, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=False)
        self.block2_norm5 = BatchNorm2d(128)
        self.block2_conv6 = Conv2d(128, 128, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=False)
        self.block2_norm6 = BatchNorm2d(128)
        self.block2_trconv7 = ConvTranspose2d(128, 256, [2, 2], [2, 2], padding=[0,0], dilation=1, groups=1, bias=False)
        self.block2_norm7 = BatchNorm2d(256)
        # block3
        self.block3_conv1 = Conv2d(128, 256, [3, 3], [2, 2], padding=[1, 1], dilation=1, groups=1, bias=False)
        self.block3_norm1 = BatchNorm2d(256)
        self.block3_conv2 = Conv2d(256, 256, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=False)
        self.block3_norm2 = BatchNorm2d(256)
        self.block3_conv3 = Conv2d(256, 256, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=False)
        self.block3_norm3 = BatchNorm2d(256)
        self.block3_conv4 = Conv2d(256, 256, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=False)
        self.block3_norm4 = BatchNorm2d(256)
        self.block3_conv5 = Conv2d(256, 256, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=False)
        self.block3_norm5 = BatchNorm2d(256)
        self.block3_conv6 = Conv2d(256, 256, [3, 3], [1, 1], padding=[1, 1], dilation=1, groups=1, bias=False)
        self.block3_norm6 = BatchNorm2d(256)
        self.block3_trconv7 = ConvTranspose2d(256, 256, [4, 4], [4, 4], padding=[0,0], dilation=1, groups=1, bias=False)
        self.block3_norm7 = BatchNorm2d(256)

        # head
        self.drop_block = DropBlock2D(block_size=3, drop_prob=0.1)
        self.head_conv_cls = Conv2d(768, 2, [1, 1], [1, 1], padding=[0, 0], dilation=1, groups=1, bias=True)
        self.head_conv_reg = Conv2d(768, 16, [1, 1], [1, 1], padding=[0, 0], dilation=1, groups=1, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # TODO: Note InstanceNorm2d is not initialized

    def forward(self, x):
        # block1
        x = self.block1_conv1(x)
        x = self.block1_norm1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.block1_conv2(x)
        x = self.block1_norm2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.block1_conv3(x)
        x = self.block1_norm3(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.block1_conv4(x)
        x = self.block1_norm4(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x_1 = self.block1_trconv5(x)
        x_1 = self.block1_norm5(x_1)
        x_1 = F.leaky_relu(x_1, negative_slope=0.1)

        # block2
        x = self.block2_conv1(x)
        x = self.block2_norm1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.block2_conv2(x)
        x = self.block2_norm2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.block2_conv3(x)
        x = self.block2_norm3(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.block2_conv4(x)
        x = self.block2_norm4(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.block2_conv5(x)
        x = self.block2_norm5(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.block2_conv6(x)
        x = self.block2_norm6(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x_2 = self.block2_trconv7(x)
        x_2 = self.block2_norm7(x_2)
        x_2 = F.leaky_relu(x_2, negative_slope=0.1)

        # block3
        x = self.block3_conv1(x)
        x = self.block3_norm1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.block3_conv2(x)
        x = self.block3_norm2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.block3_conv3(x)
        x = self.block3_norm3(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.block3_conv4(x)
        x = self.block3_norm4(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.block3_conv5(x)
        x = self.block3_norm5(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.block3_conv6(x)
        x = self.block3_norm6(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x_3 = self.block3_trconv7(x)
        x_3 = self.block3_norm7(x_3)
        x_3 = F.leaky_relu(x_3, negative_slope=0.1)

        x_ = torch.cat([x_1, x_2, x_3], 1)
        x_ = self.drop_block(x_)
        pmap = self.head_conv_cls(x_)
        pmap = torch.sigmoid(pmap)
        rmap = self.head_conv_reg(x_)
        return pmap, rmap

class VoxelNet(nn.Module):
    def __init__(self, in_channels, out_gridsize, bool_sparse=False):
        super(VoxelNet, self).__init__()
        self.bool_sparse = bool_sparse
        self.featurenet = FeatureNet(in_channels=in_channels, out_gridsize=out_gridsize, bool_sparse=self.bool_sparse)
        self.middlelayer = SparseMiddleLayer(out_gridsize) if self.bool_sparse else MiddleLayer()
        self.rpn = RPN(bool_sparse=self.bool_sparse)

    def forward(self, x, coordinate, batch_size=None):
        if not self.bool_sparse:
            x = self.featurenet(x, coordinate)
            x = self.middlelayer(x)
            x = self.rpn(x)
            return x
        else:
            assert not batch_size is None
            feat, coord = self.featurenet(x, coordinate, batch_size)
            feat = feat.permute(0, 2, 1).squeeze()
            x = self.middlelayer(feat.cuda(), coord.cuda(), batch_size)
            x = self.rpn(x)
            return x

if __name__ == "__main__":
    import time
    import numpy as np
    num_batch = [3000, 2999]
    data = torch.randn(np.sum(num_batch), 35, 7) # [#batch, #vox, #pts, #feature]
    coordinate = torch.zeros(np.sum(num_batch), 1+3).long() # [#batch, #vox, 3(z, y, x)]
    out_gridsize = (41, 1600, 352*4) #[H(z), W(y), L(x)]
    for j in range(data.shape[0]):
        if j < np.cumsum(num_batch)[0]:
            coordinate[j, :] = torch.LongTensor([0, torch.randint(out_gridsize[0], size=(1,)),
                                                    torch.randint(out_gridsize[1], size=(1,)),
                                                    torch.randint(out_gridsize[2], size=(1,))])
        elif j < np.cumsum(num_batch)[1]:
            coordinate[j, :] = torch.LongTensor([1, torch.randint(out_gridsize[0], size=(1,)),
                                                    torch.randint(out_gridsize[1], size=(1,)),
                                                    torch.randint(out_gridsize[2], size=(1,))])
    print(coordinate[num_batch[0]-1:num_batch[0]+1, :])
    sp_voxlenet = VoxelNet(in_channels=7, out_gridsize=out_gridsize, bool_sparse=True).cuda()
    t1 = time.time()
    pmap, rmap = sp_voxlenet(data.cuda(), coordinate.cuda(), batch_size=2)
    print(time.time()-t1)
    print(pmap.shape)
    print(rmap.shape)
    print("DONE")

    # import time
    # data = torch.randn(1, 3000, 35, 7) # [#batch, #vox, #pts, #feature]
    # coordinate = torch.zeros(1, 3000, 3).long() # [#batch, #vox, 3(z, y, x)]
    # out_gridsize = (10, 400, 352) #[H(z), W(y), L(x)]
    # for i in range(data.shape[0]):
    #     for j in range(data.shape[1]):
    #         coordinate[i, j, :] = torch.LongTensor([torch.randint(out_gridsize[0], size=(1,)),
    #                                                 torch.randint(out_gridsize[1], size=(1,)),
    #                                                 torch.randint(out_gridsize[2], size=(1,))])
    # voxlenet = VoxelNet(in_channels=7, out_gridsize=out_gridsize, bool_sparse=False).cuda()
    # t1 = time.time()
    # pmap, rmap = voxlenet(data.cuda(), coordinate.cuda())
    # print(time.time() - t1)
    # print(pmap.shape)
    # print(rmap.shape)
    # input()
    # print("DONE")