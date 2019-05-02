'''
File Created: Friday, 19th April 2019 2:14:03 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
from easydict import EasyDict as edict
__C = edict()
cfg = __C

__C.TAG = 'VoxelNet-000B'
__C.cls = 'Car'
__C.DATADIR = '/usr/app/data/KITTI/'
__C.gpu = 0
__C.resume = '/usr/app/det3/methods/voxelnet/saved_weights/VoxelNet-000B/140.pth.tar'
__C.start_epoch = 0
__C.epochs = 150
__C.lr = 1e-3
# __C.momentum = 0.9
__C.weight_decay = 0
__C.batch_size = 1
__C.seed = 123
__C.alpha = 1
__C.beta = 15
__C.eta = 15
__C.gamma = 0

__C.voxel_point_count = 35
__C.print_freq = 1
__C.x_range = (0, 70.4)            # Lidar Frame
__C.y_range = (-40, 40)          # Lidar Frame
__C.z_range = (-3, 1)        # Lidar Frame
__C.resolution = (0.2, 0.2, 0.4) # Lidar Frame (dx, dy, dz)
__C.cls = 'Car'
__C.KITTI_cls = {
    'Car': ['Car', 'Van'],
    'Pedestrian': ['Pedestrian'],
    'Cyclist': ['Cyclist']
    }
__C.INPUT_WIDTH = int((__C.x_range[1] - __C.x_range[0]) / __C.resolution[0])
__C.INPUT_HEIGHT = int((__C.y_range[1] - __C.y_range[0]) / __C.resolution[1])
__C.FEATURE_RATIO = 2
__C.FEATURE_WIDTH = int(__C.INPUT_WIDTH / __C.FEATURE_RATIO)
__C.FEATURE_HEIGHT = int(__C.INPUT_HEIGHT / __C.FEATURE_RATIO)
if __C.cls == 'Car':
    # car anchor
    __C.ANCHOR_L = 3.9
    __C.ANCHOR_W = 1.6
    __C.ANCHOR_H = 1.56
    __C.ANCHOR_Z = -1.0 - cfg.ANCHOR_H/2
    __C.RPN_POS_IOU = 0.6
    __C.RPN_NEG_IOU = 0.45
__C.RPN_SCORE_THRESH = 0.8 #0.96
__C.RPN_NMS_THRESH = 0.25