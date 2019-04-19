'''
File Created: Friday, 19th April 2019 2:11:32 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import os
import sys
sys.path.append('../')
import numpy as np
import torch
from torch.utils.data import Dataset
from det3.dataloarder.data import KittiData
from det3.methods.voxelnet.utils import *

class KittiDatasetVoxelNet(Dataset):
    '''
    Dataset Loader for 3D FCN
    '''
    def __init__(self, data_dir, train_val_flag, cfg):
        self.data_dir = os.path.join(data_dir, train_val_flag)
        self.train_val_flag = train_val_flag
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.image2_dir = os.path.join(self.data_dir, 'image_2')
        self.label2_dir = os.path.join(self.data_dir, 'label_2')
        self.velodyne_dir = os.path.join(self.data_dir, 'velodyne')
        self.cfg = cfg
        self.cls = cfg.cls
        self.idx_list = [itm.split('.')[0] for itm in os.listdir(self.label2_dir)]
        assert os.path.isdir(data_dir)
        assert train_val_flag in ['train', 'val', 'dev']
        assert (len(os.listdir(self.calib_dir)) == len(os.listdir(self.image2_dir))
                == len(os.listdir(self.label2_dir)) == len(os.listdir(self.velodyne_dir)))

    def __len__(self):
        return len(os.listdir(self.velodyne_dir))
    def __getitem__(self, idx):
        calib, img, label, pc = KittiData(self.data_dir, self.idx_list[idx]).read_data()
        tag = int(self.idx_list[idx])
        pc = filter_camera_angle(pc)
        voxel_dict = voxelize_pc(pc, res=self.cfg.resolution,
                                 x_range=self.cfg.x_range,
                                 y_range=self.cfg.y_range,
                                 z_range=self.cfg.z_range,
                                 num_pts_in_vox=35)
        label = filter_label_cls(label, self.cfg.KITTI_cls[self.cls])
        label = filter_label_range(label, calib, x_range=self.cfg.x_range,
                                   y_range=self.cfg.y_range, z_range=self.cfg.z_range)
        anchors = create_anchors(x_range=self.cfg.x_range,
                                 y_range=self.cfg.y_range,
                                 target_shape=(self.cfg.FEATURE_HEIGHT, self.cfg.FEATURE_WIDTH),
                                 anchor_z=self.cfg.ANCHOR_Z,
                                 anchor_size=(self.cfg.ANCHOR_L, self.cfg.ANCHOR_W, self.cfg.ANCHOR_H))
        # TODO: Test create_anchors
        return tag, pc, img, calib, label

if __name__ == "__main__":
    from det3.methods.voxelnet.config import cfg
    dataset = KittiDatasetVoxelNet(data_dir='/usr/app/data/KITTI', train_val_flag='dev', cfg=cfg)
    dataset[0]
