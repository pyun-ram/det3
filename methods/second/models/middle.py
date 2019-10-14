import torch
import torch.nn as nn
import spconv
import numpy as np

REGISTERED_MID_CLASSES = {}
def register_middle(cls, name=None):
    '''Source: https://github.com/traveller59/second.pytorch'''
    global REGISTERED_MID_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_MID_CLASSES, f"exist class: {REGISTERED_MID_CLASSES}"
    REGISTERED_MID_CLASSES[name] = cls
    return cls
def get_middle_class(name):
    '''Source: https://github.com/traveller59/second.pytorch'''
    global REGISTERED_MID_CLASSES
    assert name in REGISTERED_MID_CLASSES, f"available class: {REGISTERED_MID_CLASSES}"
    return REGISTERED_MID_CLASSES[name]

@register_middle
class SpMiddleFHD(nn.Module):
    '''Source: https://github.com/traveller59/second.pytorch'''
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 name='SpMiddleFHD'):
        super(SpMiddleFHD, self).__init__()
        from det3.methods.second.utils.torch_utils import change_default_args, Empty
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            nn.ReLU(),
            SubMConv3d(16, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            nn.ReLU(),
            SpConv3d(16, 32, 3, 2,
                     padding=1),  # [1600, 1200, 41] -> [800, 600, 21]
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.ReLU(),
            SpConv3d(32, 64, 3, 2,
                     padding=1),  # [800, 600, 21] -> [400, 300, 11]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, 3, 2,
                     padding=[0, 1, 1]),  # [400, 300, 11] -> [200, 150, 5]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1),
                     (2, 1, 1)),  # [200, 150, 5] -> [200, 150, 2]
            BatchNorm1d(64),
            nn.ReLU(),
        )
        self.max_batch_size = 6

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