'''
File Created: Thursday, 28th March 2019 4:43:02 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
from easydict import EasyDict as edict
__C = edict()
cfg = __C

__C.TAG = 'DEV'
__C.DATADIR = '/usr/app/data/KITTI/dev/'
__C.seed = None
__C.gpu = None
__C.lr = 1e-4
__C.momentum = 0.9
__C.weight_decay = 1e-4
__C.resume = None
__C.start_epoch = 0
__C.epochs = 50
__C.x_range = (0, 100)            # Lidar Frame
__C.y_range = (-40, 40)          # Lidar Frame
__C.z_range = (-2.5, 1.5)        # Lidar Frame
__C.resolution = (0.1, 0.1, 0.1) # Lidar Frame (dx, dy, dz)
__C.scale = 4
__C.cls = 'Car'

__C.KITTI_cls = {
    'Car': ['Car', 'Van'],
    'Pedestrian': ['Pedestrian'],
    'Cyclist': ['Cyclist']
    }