'''
File Created: Friday, 29th March 2019 3:39:39 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from det3.dataloarder.carladata import CarlaData
from det3.methods.fcn3d.utils import *

class CarlaDataFCN3D():
    def __init__(self,data_dir, cfg, batch_size=4, num_workers=1,distributed=False):
        self.carla_datasets = {
            x:CarlaDatasetFCN3D(data_dir=data_dir, train_val_flag=x, cfg=cfg) for x in ["train","val", "dev"]}

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.carla_datasets["train"]
                )
        else:
            train_sampler = None

        self.carla_loaders = {
            "train": torch.utils.data.DataLoader(
                self.carla_datasets["train"],
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                sampler=train_sampler,
                shuffle=True
            ),
            "val": torch.utils.data.DataLoader(
                self.carla_datasets["val"],
                batch_size=1,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=False
            ),
            "dev": torch.utils.data.DataLoader(
                self.carla_datasets["dev"],
                batch_size=1,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=False
            )}


class CarlaDatasetFCN3D(Dataset):
    '''
    Dataset Loader for 3D FCN (CARLA)
    '''
    def __init__(self, data_dir, train_val_flag, cfg):
        self.data_dir = os.path.join(data_dir, train_val_flag)
        self.train_val_flag = train_val_flag
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_imu')
        self.cfg = cfg
        self.cls = cfg.cls
        self.idx_list = [itm.split('.')[0] for itm in os.listdir(self.label_dir)]
        self.idx_list.sort()
        assert os.path.isdir(data_dir)
        assert train_val_flag in ['train', 'val', 'dev']
        assert len(os.listdir(self.calib_dir)) == len(os.listdir(self.label_dir))

    def __len__(self):
        return len(os.listdir(self.label_dir))

    def __getitem__(self, idx):
        pc_dict, label, calib = CarlaData(self.data_dir, self.idx_list[idx]).read_data()
        tag = int(self.idx_list[idx])
        pc = calib.lidar2imu(pc_dict['velo_top'], key='Tr_imu_to_velo_top')
        pc = filter_camera_angle(pc)
        voxel = voxelize_pc(pc, res=self.cfg.resolution, x_range=self.cfg.x_range,
                            y_range=self.cfg.y_range, z_range=self.cfg.z_range)
        label = filter_label_cls(label, self.cfg.KITTI_cls[self.cls])
        label = filter_label_range(label, calib, x_range=self.cfg.x_range,
                                   y_range=self.cfg.y_range, z_range=self.cfg.z_range)
        label = filter_label_pts(label, calib, pc, threshold=20)
        gt_objgrid = create_objectgrid(label, calib,
                                       res=tuple([self.cfg.scale * _d for _d in self.cfg.resolution]),
                                       x_range=self.cfg.x_range,
                                       y_range=self.cfg.y_range,
                                       z_range=self.cfg.z_range)
        gt_reggrid = create_regressgrid(label, calib,
                                        res=tuple([self.cfg.scale * _d for _d in self.cfg.resolution]),
                                        x_range=self.cfg.x_range,
                                        y_range=self.cfg.y_range,
                                        z_range=self.cfg.z_range)
        # rec_label = parse_grid_to_label(gt_objgrid, gt_reggrid, score_threshold=0.8, nms_threshold=0.1, calib=calib, cls='Car',
        #                                 res=tuple([self.cfg.scale * _d for _d in self.cfg.resolution]),
        #                                 x_range=self.cfg.x_range,
        #                                 y_range=self.cfg.y_range,
        #                                 z_range=self.cfg.z_range)
        # rec_bool = label.equal(rec_label, acc_cls=cfg.KITTI_cls[self.cls], rtol=0.1)
        # return voxel, gt_objgrid, gt_reggrid, rec_bool, label, rec_label
        voxel = np.expand_dims(voxel, axis=0).astype(np.float32)
        gt_objgrid = gt_objgrid.astype(np.float32)
        gt_reggrid = gt_reggrid.astype(np.float32)
        if self.train_val_flag in ['train', 'dev']:
            return tag, voxel, gt_objgrid, gt_reggrid
        elif self.train_val_flag in ['val']:
            voxel = np.expand_dims(voxel, axis=0).astype(np.float32)
            gt_objgrid = np.expand_dims(gt_objgrid, axis=0).astype(np.float32)
            gt_reggrid = np.expand_dims(gt_reggrid, axis=0).astype(np.float32)
            return tag, voxel, None, gt_objgrid, gt_reggrid, pc, label, calib
        else:
            raise NotImplementedError

if __name__ == '__main__':
    from det3.methods.fcn3d.config import cfg
    dataloader = CarlaDataFCN3D(data_dir='/usr/app/data/CARLA/', cfg=cfg, batch_size=5).carla_loaders
    train_loader = dataloader['train']
    for i, (tag, voxel, gt_objgrid, gt_reggrid) in enumerate(train_loader):
        print(voxel.dtype)
        print(gt_objgrid.dtype)
        print(gt_reggrid.dtype)