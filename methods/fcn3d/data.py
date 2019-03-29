'''
File Created: Friday, 29th March 2019 3:39:39 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''

import os
import sys
sys.path.append('../')
from torch.utils.data import Dataset
from det3.dataloarder.data import KittiData
from det3.methods.fcn3d.utils import filter_camera_angle, pc_to_voxel

class KittiDataFCN3D(Dataset):
    '''
    Dataset Loader for 3D FCN
    '''
    def __init__(self, data_dir, train_val_flag, cfg, cls='Car'):
        self.data_dir = os.path.join(data_dir, train_val_flag)
        self.train_val_flag = train_val_flag
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.image2_dir = os.path.join(self.data_dir, 'image_2')
        self.label2_dir = os.path.join(self.data_dir, 'label_2')
        self.velodyne_dir = os.path.join(self.data_dir, 'velodyne')
        self.cfg = cfg
        self.idx_list = [itm.split('.')[0] for itm in os.listdir(self.label2_dir)]
        assert os.path.isdir(data_dir)
        assert train_val_flag in ['train', 'val', 'dev']
        assert (len(os.listdir(self.calib_dir)) == len(os.listdir(self.image2_dir))
                == len(os.listdir(self.label2_dir)) == len(os.listdir(self.velodyne_dir)))

    def __len__(self):
        return len(os.listdir(self.velodyne_dir))

    
    def __getitem__(self, idx):
        calib, _, label, pc = KittiData(self.data_dir, self.idx_list[idx]).read_data()
        pc = filter_camera_angle(pc)
        voxel = pc_to_voxel(pc, res=self.cfg.resolution, x=self.cfg.x_range, y=self.cfg.y_range, z=self.cfg.z_range)
        

if __name__ == '__main__':
    from det3.methods.fcn3d.config import cfg
    dataset = KittiDataFCN3D(data_dir='/usr/app/data/KITTI', train_val_flag='dev', cfg=cfg)
    print(len(dataset))
    dataset[0]
