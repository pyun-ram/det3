'''
File Created: Friday, 19th April 2019 2:14:03 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
from easydict import EasyDict as edict
import math
import numpy as np
__C = edict()
cfg = __C

__C.TAG = 'VoxelNet-dev-000A'
__C.cls = 'Car'
__C.DATADIR = '/usr/app/data/KITTI/'
__C.gpu = 0
__C.resume = None
__C.start_epoch = 0
__C.epochs = 200
__C.lr_dict = {
    "mode": "super-converge", # "const", "decay", "super-converge"
    "lr_range": [1e-4, 1e-3],
    "lr": 1e-3,
    "cycle": 9, # epochs for one cycle
}
# __C.lr_dict = {
#     "mode": "const", # "const", "decay", "super-converge"
#     "lr": 1e-3,
# }
# __C.lr_dict = {
#     "mode": "decay", # "const", "decay", "super-converge"
#     "lr": 1e-3,
#     "cycle": 30, #  epochs for one decay
#     "factor": 0.1,
# }
# __C.momentum = 0.9

__C.weight_decay = 0
__C.batch_size = 8
__C.num_trainloader_wkers = 8
__C.seed = None
__C.alpha = 1
__C.beta = 1
__C.eta = 0.5
__C.gamma = 2.0
__C.lambda_rot = 0 # weight of rot regularization term
__C.weight_var = 1
__C.sparse = True
__C.bool_fast_loader = False
__C.val_freq = 5
__C.val_max_visnum = 100
__C.cls = 'Car'
__C.name_featurenet = "SimpleVoxel" #FeatureNet" or "SimpleVoxel"
__C.name_RPN = "RPNV3" #"RPN" or "RPNV2"
__C.KITTI_cls = {
    'Car': ['Car', 'Van'],
    'Pedestrian': ['Pedestrian'],
    'Cyclist': ['Cyclist']
    }
__C.aug_dict = {
    "p_rot":0.1,
    "p_tr": 0.1,
    "p_flip": 0.1,
    "p_keep": 0.7
}
__C.aug_param = {
    "dx_range": [-0.5, 0.5],
    "dy_range": [-0.5, 0.5],
    "dz_range": [-0.2, 0.2],
    "dry_range": [-5 / 180.0 * np.pi, 5 / 180.0 * np.pi]
}
if __C.sparse:
    __C.FEATURE_RATIO = 8
    __C.voxel_point_count = 5
    __C.print_freq = 1
    __C.x_range = (0, 70.4)            # Lidar Frame
    __C.y_range = (-40, 40)          # Lidar Frame
    __C.z_range = (-3, 1.1)        # Lidar Frame
    __C.resolution = (0.05, 0.05, 0.1) # Lidar Frame (dx, dy, dz)
else:
    __C.FEATURE_RATIO = 2
    __C.voxel_point_count = 35
    __C.print_freq = 1
    __C.x_range = (0, 70.4)            # Lidar Frame
    __C.y_range = (-40, 40)          # Lidar Frame
    __C.z_range = (-3, 1)        # Lidar Frame
    __C.resolution = (0.2, 0.2, 0.4) # Lidar Frame (dx, dy, dz)

__C.INPUT_WIDTH = int((__C.x_range[1] - __C.x_range[0]) / __C.resolution[0])
__C.INPUT_HEIGHT = int((__C.y_range[1] - __C.y_range[0]) / __C.resolution[1])
__C.FEATURE_WIDTH = int(__C.INPUT_WIDTH / __C.FEATURE_RATIO)
__C.FEATURE_HEIGHT = int(__C.INPUT_HEIGHT / __C.FEATURE_RATIO)
if __C.cls == 'Car':
    # car anchor
    __C.ANCHOR_L = 3.9
    __C.ANCHOR_W = 1.6
    __C.ANCHOR_H = 1.56
    __C.ANCHOR_Z = -1.0 - __C.ANCHOR_H/2
    __C.RPN_POS_IOU = 0.6
    __C.RPN_NEG_IOU = 0.45
__C.RPN_SCORE_THRESH = 0.8 #0.96
__C.RPN_NMS_THRESH = 0.01
__C.MIDGRID_SHAPE = [int(itm) for itm in [math.ceil((__C.z_range[1] - __C.z_range[0])/__C.resolution[-1]),
                                          __C.INPUT_HEIGHT,
                                          __C.INPUT_WIDTH]]
__C.log_grad = False
__C.log_actv = False