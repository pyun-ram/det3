'''
File Created: Wednesday, 26th June 2019 10:25:13 am
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import sys
sys.path.append("../")
from det3.dataloarder.kittidata import KittiLabel, KittiCalib
from det3.utils.utils import istype, apply_R, apply_tr, rotz
import numpy as np
from typing import List


class KittiAugmentor:
    def __init__(self):
        self.dataset = 'Kitti'

    def rotate_obj(self, label: KittiLabel, pc: np.array, calib: KittiCalib, dry_range: List[float]) -> (KittiLabel, np.array):
        '''
        rotate object along the z axis in the LiDAR frame
        inputs:
            label: gt
            pc: [#pts, >= 3]
            calib:
            dry_range: [dry_min, dry_max] in radius
        returns:
            label_rot
            pc_rot
        Note: The inputs (label and pc) are not safe
        '''
        assert istype(label, "KittiLabel")
        dry_min, dry_max = dry_range
        for obj in label.data:
            dry = np.random.rand() * (dry_max - dry_min) + dry_min
            # modify pc
            idx = obj.get_pts_idx(pc[:, :3], calib)
            bottom_Fcam = np.array([obj.x, obj.y, obj.z]).reshape(1, -1)
            bottom_Flidar = calib.leftcam2lidar(bottom_Fcam)
            pc[idx, :3] = apply_tr(pc[idx, :3], -bottom_Flidar)
            # obj.ry += dry is correspond to rotz(-dry)
            # since obj is in cam frame
            # pc is in LiDAR frame
            pc[idx, :3] = apply_R(pc[idx, :3], rotz(-dry))
            pc[idx, :3] = apply_tr(pc[idx, :3], bottom_Flidar)
            # modify obj
            obj.ry += dry
        return label, pc

    def tr_obj(self, label: KittiLabel, pc: np.array, calib: KittiCalib, 
        dx_range: List[float], dy_range: List[float], dz_range: List[float]) -> (KittiLabel, np.array):
        '''
        translate object in the LiDAR frame
        inputs:
            label: gt
            pc: [#pts, >= 3]
            calib:
            dx_range: [dx_min, dx_max]
            dy_range: [dy_min, dy_max]
            dz_range: [dz_min, dz_max]
        returns:
            label_tr
            pc_tr
        Note: The inputs (label and pc) are not safe
        '''
        assert istype(label, "KittiLabel")
        dx_min, dx_max = dx_range
        dy_min, dy_max = dy_range
        dz_min, dz_max = dz_range
        for obj in label.data:
            dx = np.random.rand() * (dx_max - dx_min) + dx_min
            dy = np.random.rand() * (dy_max - dy_min) + dy_min
            dz = np.random.rand() * (dz_max - dz_min) + dz_min
            # modify pc
            idx = obj.get_pts_idx(pc[:, :3], calib)
            dtr = np.array([dx, dy, dz]).reshape(1, -1)
            pc[idx, :3] = apply_tr(pc[idx, :3], dtr)
            # modify obj
            bottom_Fcam = np.array([obj.x, obj.y, obj.z]).reshape(1, -1)
            bottom_Flidar = calib.leftcam2lidar(bottom_Fcam)
            bottom_Flidar = apply_tr(bottom_Flidar, dtr)
            bottom_Fcam = calib.lidar2leftcam(bottom_Flidar)
            obj.x, obj.y, obj.z = bottom_Fcam.flatten()
        return label, pc

    def flip_pc(self, label: KittiLabel, pc: np.array, calib: KittiCalib) -> (KittiLabel, np.array):
        '''
        flip point cloud along the y axis of the Kitti Lidar frame
        inputs:
            label: ground truth
            pc: point cloud
            calib:
        Note: The inputs (label and pc) are not safe
        '''
        assert istype(label, "KittiLabel")
        # flip point cloud
        pc[:, 1] *= -1
        # modify gt
        for obj in label.data:
            bottom_Fcam = np.array([obj.x, obj.y, obj.z]).reshape(1, -1)
            bottom_Flidar = calib.leftcam2lidar(bottom_Fcam)
            bottom_Flidar[0, 1] *= -1
            bottom_Fcam = calib.lidar2leftcam(bottom_Flidar)
            obj.x, obj.y, obj.z = bottom_Fcam.flatten()
            obj.ry *= -1
        return label, pc

if __name__ == "__main__":
    from det3.dataloarder.kittidata import KittiData
    from det3.visualizer.vis import BEVImage, FVImage
    from PIL import Image
    import os
    from det3.utils.utils import get_idx_list
    idx_list = get_idx_list("/usr/app/data/KITTI/split_index/train.txt")
    for idx in idx_list:
        print(idx)
        calib, img, label, pc = KittiData("/usr/app/data/KITTI/train", idx).read_data()

        bevimg = BEVImage(x_range=(0, 70), y_range=(-40, 40), grid_size=(0.05, 0.05))
        bevimg.from_lidar(pc, scale=1)
        fvimg = FVImage()
        fvimg.from_lidar(calib, pc[:, :3])
        for obj in label.data:
            bevimg.draw_box(obj, calib, bool_gt=True)
            fvimg.draw_box(obj, calib, bool_gt=True)
        bevimg_img = Image.fromarray(bevimg.data)
        bevimg_img.save(os.path.join('/usr/app/vis/train', idx+'.png'))
        fvimg_img = Image.fromarray(fvimg.data)
        fvimg_img.save(os.path.join('/usr/app/vis/train', idx+'fv.png'))
        kitti_agmtor = KittiAugmentor()
        label, pc = kitti_agmtor.tr_obj(label, pc, calib, dx_range = [-0.25, 0.25], dy_range = [-0.25, 0.25], dz_range = [-0.1, 0.1])
        bevimg = BEVImage(x_range=(0, 70), y_range=(-40, 40), grid_size=(0.05, 0.05))
        bevimg.from_lidar(pc, scale=1)
        fvimg = FVImage()
        fvimg.from_lidar(calib,pc[:, :3])
        for obj in label.data:
            bevimg.draw_box(obj, calib, bool_gt=True)
            fvimg.draw_box(obj, calib, bool_gt=True)
        bevimg_img = Image.fromarray(bevimg.data)
        bevimg_img.save(os.path.join('/usr/app/vis/train', idx+'_aug.png'))
        fvimg_img = Image.fromarray(fvimg.data)
        fvimg_img.save(os.path.join('/usr/app/vis/train', idx+'_augfv.png'))
