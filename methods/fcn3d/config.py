'''
File Created: Thursday, 28th March 2019 4:43:02 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
from easydict import EasyDict as edict
__C = edict()
cfg = __C

__C.TAG = 'FCN-CARLA-000A'
__C.DATADIR = '/usr/app/data/CARLA/'
__C.seed = None
__C.gpu = 0
__C.lr = 0.001
__C.momentum = 0.9
__C.weight_decay = 1e-4
__C.resume = None
__C.start_epoch = 0
__C.epochs = 500
__C.x_range = (0, 80)            # Lidar Frame / IMU Frame
__C.y_range = (-40, 40)          # Lidar Frame / IMU Frame
__C.z_range = (-2.5, 1.5)        # Lidar Frame / IMU Frame
__C.resolution = (0.1, 0.1, 0.1) # Lidar Frame (dx, dy, dz) / IMU Frame
__C.scale = 4
__C.cls = 'Car'
__C.print_freq = 1
__C.threshold = 0.99
__C.alpha = 1
__C.beta = 5
__C.eta = 10
__C.gamma = 0
__C.nms_threshold = 0.1
__C.batch_size = 2

__C.KITTI_cls = {
    'Car': ['Car', 'Van'],
    'Pedestrian': ['Pedestrian'],
    'Cyclist': ['Cyclist']
    }