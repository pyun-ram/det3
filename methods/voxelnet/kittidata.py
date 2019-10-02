'''
File Created: Friday, 19th April 2019 2:11:32 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from det3.dataloader.kittidata import KittiData
from det3.methods.voxelnet.utils import *
from det3.dataloader.augmentor import KittiAugmentor
from det3.utils.utils import load_pickle
import glob

def collate_fn(batch):
    num_batch = len(batch)
    tag, voxel_feature, coordinate, gt_pos_map, gt_neg_map, gt_target_map = [], [], [], [], [], []
    for i in range(num_batch):
        tag.append(batch[i][0])
        voxel_feature.append(batch[i][1])
        tmp_coordinate = np.hstack([np.ones((batch[i][2].shape[0], 1))*i,
                                    batch[i][2]])
        coordinate.append(tmp_coordinate)
        tmp_gt_pos_map = np.expand_dims(batch[i][3], axis=0)
        gt_pos_map.append(tmp_gt_pos_map)
        tmp_gt_neg_map = np.expand_dims(batch[i][4], axis=0)
        gt_neg_map.append(tmp_gt_neg_map)
        tmp_gt_target_map = np.expand_dims(batch[i][5], axis=0)
        gt_target_map.append(tmp_gt_target_map)
    tag = tag
    voxel_feature = torch.from_numpy(np.vstack(voxel_feature)).contiguous()
    coordinate = torch.from_numpy(np.vstack(coordinate)).contiguous()
    gt_pos_map = torch.from_numpy(np.vstack(gt_pos_map)).contiguous()
    gt_neg_map = torch.from_numpy(np.vstack(gt_neg_map)).contiguous()
    gt_target_map = torch.from_numpy(np.vstack(gt_target_map)).contiguous()
    return tag, voxel_feature, coordinate, gt_pos_map, gt_neg_map, gt_target_map

class KITTIDataVoxelNet():
    def __init__(self, data_dir, cfg, batch_size=4, num_workers=1, distributed=False):
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
                shuffle=True,
                collate_fn=collate_fn
            ),
            "dev": torch.utils.data.DataLoader(
                self.kitti_datasets["dev"],
                batch_size=1,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=False
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
        self.idx_list = [itm.split('.')[0] for itm in os.listdir(self.calib_dir)]
        self.idx_list.sort()
        assert os.path.isdir(data_dir)
        assert train_val_flag in ['train', 'val', 'dev', 'test']
        if train_val_flag  in ['train', 'val', 'dev']:
            assert (len(os.listdir(self.calib_dir)) == len(os.listdir(self.image2_dir))
                    == len(os.listdir(self.label2_dir)) == len(os.listdir(self.velodyne_dir)))
        if self.train_val_flag in ['val', 'dev']:
            self.bool_fast_loader = cfg.bool_fast_loader
            self.fastload_dir = os.path.join(self.data_dir, 'fast_load') if self.bool_fast_loader else None
            if self.bool_fast_loader and not os.path.isdir(self.fastload_dir):
                print("ERROR: Fast Loading mode, but the fast_load dir is not available.")
                raise RuntimeError
        else:
            self.bool_fast_loader = cfg.bool_fast_loader
            self.num_train_fast_load = len(glob.glob(os.path.join(self.data_dir, "fast_load_*")))
            self.fastload_dir = None # wait for update

    def __len__(self):
        return len(os.listdir(self.velodyne_dir))

    def __getitem__(self, idx):
        if self.bool_fast_loader and self.train_val_flag in ['dev', 'val']:
            return load_pickle(os.path.join(self.fastload_dir, "{}.pkl".format(self.idx_list[idx])))
        elif self.bool_fast_loader and self.train_val_flag == 'train':
            self.fastload_dir = os.path.join(self.data_dir, "fast_load_{}"
                                             .format(np.random.randint(self.num_train_fast_load)))
            return load_pickle(os.path.join(self.fastload_dir, "{}.pkl".format(self.idx_list[idx])))
        if self.train_val_flag in ['train', 'val', 'dev']:
            kittidata_output_dict = None
        elif self.train_val_flag == "test":
            bool_have_label = os.path.isdir(self.label2_dir)
            kittidata_output_dict = {"calib": True,
                                     "image": True,
                                     "label": bool_have_label,
                                     "velodyne": True}
        else:
            raise NotImplementedError
        calib, img, label, pc = KittiData(self.data_dir, self.idx_list[idx], output_dict=kittidata_output_dict).read_data()
        tag = int(self.idx_list[idx])
        pc = filter_camera_angle(pc)
        if self.train_val_flag in ["train"]:
            agmtor = KittiAugmentor(p_rot=self.cfg.aug_dict["p_rot"],
                                    p_tr=self.cfg.aug_dict["p_tr"],
                                    p_flip=self.cfg.aug_dict["p_flip"],
                                    p_keep=self.cfg.aug_dict["p_keep"],
                                    dx_range=self.cfg.aug_param["dx_range"],
                                    dy_range=self.cfg.aug_param["dy_range"],
                                    dz_range=self.cfg.aug_param["dz_range"],
                                    dry_range=self.cfg.aug_param["dry_range"])
            label, pc = agmtor.apply(label, pc, calib)
        voxel_dict = voxelize_pc(pc, res=self.cfg.resolution,
                                 x_range=self.cfg.x_range,
                                 y_range=self.cfg.y_range,
                                 z_range=self.cfg.z_range,
                                 num_pts_in_vox=self.cfg.voxel_point_count)
        voxel_feature = voxel_dict["feature_buffer"].astype(np.float32)
        coordinate = voxel_dict["coordinate_buffer"].astype(np.int64)
        anchors = create_anchors(x_range=self.cfg.x_range,
                                 y_range=self.cfg.y_range,
                                 target_shape=(self.cfg.FEATURE_HEIGHT, self.cfg.FEATURE_WIDTH),
                                 anchor_z=self.cfg.ANCHOR_Z,
                                 anchor_size=(self.cfg.ANCHOR_L, self.cfg.ANCHOR_W, self.cfg.ANCHOR_H))
        if self.train_val_flag == "test":
            coordinate = np.hstack([np.zeros((coordinate.shape[0], 1)),
                                    coordinate])
            return tag, voxel_feature, coordinate, img, anchors, pc, label, calib
        label = filter_label_cls(label, self.cfg.KITTI_cls[self.cls])
        label = filter_label_range(label, calib, x_range=self.cfg.x_range,
                                   y_range=self.cfg.y_range, z_range=self.cfg.z_range)
        label = filter_label_pts(label, pc[:, :3], calib, threshold_pts=10)
        gt_pos_map, gt_neg_map, gt_target = create_rpn_target(label, calib,
                                                              target_shape=(self.cfg.FEATURE_HEIGHT, self.cfg.FEATURE_WIDTH),
                                                              anchors=anchors, threshold_pos_iou=self.cfg.RPN_POS_IOU,
                                                              threshold_neg_iou=self.cfg.RPN_NEG_IOU, anchor_size=(self.cfg.ANCHOR_L, self.cfg.ANCHOR_W, self.cfg.ANCHOR_H))
        gt_pos_map = gt_pos_map.astype(np.float32)
        gt_pos_map = np.transpose(gt_pos_map, (2, 0, 1))
        gt_neg_map = gt_neg_map.astype(np.float32)
        gt_neg_map = np.transpose(gt_neg_map, (2, 0, 1))
        gt_target = gt_target.astype(np.float32)
        gt_target = np.transpose(gt_target, (2, 0, 1))
        if self.train_val_flag in ['train']:
            return tag, voxel_feature, coordinate, gt_pos_map, gt_neg_map, gt_target
        elif self.train_val_flag in ['val', 'dev']:
            coordinate = np.hstack([np.zeros((coordinate.shape[0], 1)),
                                    coordinate])
            gt_pos_map = np.expand_dims(gt_pos_map, 0)
            gt_neg_map = np.expand_dims(gt_neg_map, 0)
            gt_target = np.expand_dims(gt_target, 0)
            return tag, voxel_feature, coordinate, gt_pos_map, gt_neg_map, gt_target, anchors, pc, label, calib
        else:
            raise NotImplementedError

if __name__ == "__main__":
    from det3.methods.voxelnet.config import cfg
    dataset = KittiDatasetVoxelNet(data_dir='/usr/app/data/KITTI', train_val_flag='train', cfg=cfg)
    kitti_data = KITTIDataVoxelNet(data_dir=cfg.DATADIR, cfg=cfg, batch_size=5).kitti_loaders
    train_loader = kitti_data["train"]
    # dataset[2835] # 4448
    # dataset[48]
    for i,  (tag, voxel_feature, coordinate, gt_pos_map, gt_neg_map, gt_target) in enumerate(train_loader):
        print(tag)
        print(voxel_feature.shape)
        print(coordinate.shape)
        print(gt_pos_map.shape)
        print(gt_neg_map.shape)
        print(gt_target.shape)
        input()

