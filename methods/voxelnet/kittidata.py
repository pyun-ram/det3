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
from det3.dataloarder.kittidata import KittiData
from det3.methods.voxelnet.utils import *
from det3.dataloarder.augmentor import KittiAugmentor

class KITTIDataVoxelNet():
    def __init__(self,data_dir, cfg, batch_size=4, num_workers=1,distributed=False):
        """
        Dataset Loader for VoxelNet (train and dev)
        """
        self.kitti_datasets = {
            x:KittiDatasetVoxelNet(data_dir=data_dir, train_val_flag=x, cfg=cfg) for x in ["train", "dev"]}

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
            "dev": torch.utils.data.DataLoader(
                self.kitti_datasets["dev"],
                batch_size=1,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=True
            )}

class KittiDatasetVoxelNet(Dataset):
    '''
    Dataset Loader for VoxelNet
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
        self.idx_list.sort()
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
        if not self.train_val_flag == "val":
            agmtor = KittiAugmentor(p_rot=self.cfg.aug_dict["p_rot"], p_tr=self.cfg.aug_dict["p_tr"],
                                    p_flip=self.cfg.aug_dict["p_flip"],p_keep=self.cfg.aug_dict["p_keep"],
                                    dx_range=self.cfg.aug_param["dx_range"], dy_range=self.cfg.aug_param["dy_range"],
                                    dz_range=self.cfg.aug_param["dz_range"], dry_range=self.cfg.aug_param["dry_range"])
            label, pc = agmtor.apply(label, pc, calib)
        voxel_dict = voxelize_pc(pc, res=self.cfg.resolution,
                                 x_range=self.cfg.x_range,
                                 y_range=self.cfg.y_range,
                                 z_range=self.cfg.z_range,
                                 num_pts_in_vox=self.cfg.voxel_point_count)
        label = filter_label_cls(label, self.cfg.KITTI_cls[self.cls])
        label = filter_label_range(label, calib, x_range=self.cfg.x_range,
                                   y_range=self.cfg.y_range, z_range=self.cfg.z_range)
        label = filter_label_pts(label, pc[:, :3], calib, threshold_pts=10)
        anchors = create_anchors(x_range=self.cfg.x_range,
                                 y_range=self.cfg.y_range,
                                 target_shape=(self.cfg.FEATURE_HEIGHT, self.cfg.FEATURE_WIDTH),
                                 anchor_z=self.cfg.ANCHOR_Z,
                                 anchor_size=(self.cfg.ANCHOR_L, self.cfg.ANCHOR_W, self.cfg.ANCHOR_H))
        gt_pos_map, gt_neg_map, gt_target = create_rpn_target(label, calib,
                                                     target_shape=(self.cfg.FEATURE_HEIGHT, self.cfg.FEATURE_WIDTH),
                                                     anchors=anchors, threshold_pos_iou=self.cfg.RPN_POS_IOU,
                                                     threshold_neg_iou=self.cfg.RPN_NEG_IOU, anchor_size=(self.cfg.ANCHOR_L, self.cfg.ANCHOR_W, self.cfg.ANCHOR_H))
        voxel_feature = voxel_dict["feature_buffer"].astype(np.float32)
        coordinate = voxel_dict["coordinate_buffer"].astype(np.int64)
        gt_pos_map = gt_pos_map.astype(np.float32)
        gt_pos_map = np.transpose(gt_pos_map, (2, 0, 1))
        gt_neg_map = gt_neg_map.astype(np.float32)
        gt_neg_map = np.transpose(gt_neg_map, (2, 0, 1))
        gt_target = gt_target.astype(np.float32)
        gt_target = np.transpose(gt_target, (2, 0, 1))
        # rec_label = parse_grid_to_label(gt_pos_map, gt_target, anchors, 
        #                                 anchor_size=(self.cfg.ANCHOR_L, self.cfg.ANCHOR_W, self.cfg.ANCHOR_H),
        #                                 cls=self.cfg.cls, calib=calib, threshold_score=self.cfg.RPN_SCORE_THRESH,
        #                                 threshold_nms=self.cfg.RPN_NMS_THRESH)
        # rec_bool = label.equal(rec_label, acc_cls=self.cfg.KITTI_cls[self.cfg.cls], rtol=1e-5)
        # print(tag, rec_bool, len(label.data), rec_label)
        if self.train_val_flag in ['train', 'dev']:
            return tag, voxel_feature, coordinate, gt_pos_map, gt_neg_map, gt_target
        elif self.train_val_flag in ['val']:
            voxel_feature = np.expand_dims(voxel_feature, 0)
            coordinate = np.expand_dims(coordinate, 0)
            gt_pos_map = np.expand_dims(gt_pos_map, 0)
            gt_neg_map = np.expand_dims(gt_neg_map, 0)
            gt_target = np.expand_dims(gt_target, 0)
            return tag, voxel_feature, coordinate, gt_pos_map, gt_neg_map, gt_target, anchors, pc, label, calib
        else:
            raise NotImplementedError

if __name__ == "__main__":
    from det3.methods.voxelnet.config import cfg
    dataset = KittiDatasetVoxelNet(data_dir='/usr/app/data/KITTI', train_val_flag='dev', cfg=cfg)
    # dataset[2835] # 4448
    # dataset[48]
    for i in range(0, len(dataset)):
        dataset[i]
