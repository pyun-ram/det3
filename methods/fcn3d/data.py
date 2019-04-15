'''
File Created: Friday, 29th March 2019 3:39:39 pm
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
from det3.methods.fcn3d.utils import *

class KITTIDataFCN3D():
    def __init__(self,data_dir, cfg, batch_size=4, num_workers=1,distributed=False):
        """
        """

        self.kitti_datasets = {
            x:KittiDatasetFCN3D(data_dir=data_dir, train_val_flag=x, cfg=cfg) for x in ["train","val"]}

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.kitti_datasets["train"]
                )
        else:
            train_sampler = None

        self.kitti_loaders = {
            "train": torch.utils.data.DataLoader(
                self.kitti_datasets["train"],
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                sampler=train_sampler,
                shuffle=True
            ),
            "val": torch.utils.data.DataLoader(
                self.kitti_datasets["val"],
                batch_size=1,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=False
            )}

class KittiDatasetFCN3D(Dataset):
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
        calib, _, label, pc = KittiData(self.data_dir, self.idx_list[idx]).read_data()
        tag = int(self.idx_list[idx])
        pc = filter_camera_angle(pc)
        voxel = voxelize_pc(pc, res=self.cfg.resolution, x_range=self.cfg.x_range,
                            y_range=self.cfg.y_range, z_range=self.cfg.z_range)
        label = filter_label_cls(label, self.cfg.KITTI_cls[self.cls])
        label = filter_label_range(label, calib, x_range=self.cfg.x_range,
                                   y_range=self.cfg.y_range, z_range=self.cfg.z_range) # BUG HERE
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
        # rec_label = parse_grid_to_label(gt_objgrid, gt_reggrid, 0.8, calib, 'Car',
        #                                 res=tuple([self.cfg.scale * _d for _d in self.cfg.resolution]),
        #                                 x_range=self.cfg.x_range,
        #                                 y_range=self.cfg.y_range,
        #                                 z_range=self.cfg.z_range)
        # rec_bool = label.equal(rec_label, acc_cls=cfg.KITTI_cls[self.cls], rtol=0.1)
        # return voxel, gt_objgrid, gt_reggrid, rec_bool, label, rec_label
        voxel = np.expand_dims(voxel, axis=0).astype(np.float32)
        gt_objgrid = gt_objgrid.astype(np.float32)
        gt_reggrid = gt_reggrid.astype(np.float32)
        if self.train_val_flag in ['train'] :
            return tag, voxel, gt_objgrid, gt_reggrid
        elif self.train_val_flag in ['val', 'dev']:
            voxel = np.expand_dims(voxel, axis=0).astype(np.float32)
            gt_objgrid = np.expand_dims(gt_objgrid, axis=0).astype(np.float32)
            gt_reggrid = np.expand_dims(gt_reggrid, axis=0).astype(np.float32)
            return tag, voxel, gt_objgrid, gt_reggrid, pc, label, calib
        else:
            raise NotImplementedError

if __name__ == '__main__':
    from det3.methods.fcn3d.config import cfg
    # dataset = KittiDataFCN3D(data_dir='/usr/app/data/KITTI', train_val_flag='val', cfg=cfg)
    # num_true = 0
    # for i, data in enumerate(dataset):
    #     _, _, _, rec_bool, label, rec_label = data
    #     num_true = num_true + 1 if rec_bool else num_true
    #     print("{}/{}: {} ({})".format(i, len(dataset), rec_bool, len(label.data)))
    #     if not rec_bool:
    #         print(label)
    #         print(rec_label)
    #         input()
    # print("TEST DONE: {}/{} is PASS.".format(num_true, len(dataset)))
    kitti_loaders = KITTIDataFCN3D(data_dir='/usr/app/data/KITTI', cfg=cfg, batch_size=5).kitti_loaders
    train_loader = kitti_loaders["train"]
    val_loader = kitti_loaders["val"]
    dev_loader = kitti_loaders["dev"]
    print(len(train_loader))
    print(len(val_loader))
    print(len(dev_loader))
    for i, (voxel, gt_objgrid, gt_reggrid) in enumerate(dev_loader):
        print(voxel.dtype)
        print(gt_objgrid.dtype)
        print(gt_reggrid.dtype)