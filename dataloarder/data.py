'''
File Created: Sunday, 17th March 2019 3:58:52 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import os
import numpy as np
from numpy.linalg import inv
try:
    from ..utils import utils
except:
    # Run script python3 dataloader/data.py
    import sys
    sys.path.append("../")
    import det3.utils.utils as utils

# KITTI
class KittiCalib:
    '''
    class storing KITTI calib data
        self.data(None/dict):keys: 'P0', 'P1', 'P2', 'P3', 'R0_rect', 'Tr_velo_to_cam', 'Tr_imu_to_velo'
        self.R0_rect(np.array):  [4,4]
        self.Tr_velo_to_cam(np.array):  [4,4]
    '''
    def __init__(self, calib_path):
        self.path = calib_path
        self.data = None
        self.R0_rect = None
        self.Tr_velo_to_cam = None

    def read_kitti_calib_file(self):
        '''
        read KITTI calib file
        '''
        calib = dict()
        with open(self.path, 'r') as f:
            str_list = f.readlines()
        str_list = [itm.rstrip() for itm in str_list if itm != '\n']
        for itm in str_list:
            calib[itm.split(':')[0]] = itm.split(':')[1]
        for k, v in calib.items():
            calib[k] = [float(itm) for itm in v.split()]
        self.data = calib

        R0_rect = np.zeros([4, 4])
        R0_rect[0:3, 0:3] = np.array(self.data['R0_rect']).reshape(3, 3)
        R0_rect[3, 3] = 1
        self.R0_rect = R0_rect

        Tr_velo_to_cam = np.zeros([4, 4])
        Tr_velo_to_cam[0:3, :] = np.array(self.data['Tr_velo_to_cam']).reshape(3, 4)
        Tr_velo_to_cam[3, 3] = 1
        self.Tr_velo_to_cam = Tr_velo_to_cam
        return self
    
    def leftcam2lidar(self, pts):
        '''
        transform the pts from the left camera frame to lidar frame
        pts_lidar  = Tr_velo_to_cam^{-1} @ R0_rect^{-1} @ pts_cam
        inputs:
            pts(np.array): [#pts, 3]
                points in the left camera frame
        '''
        if self.data == None:
            print("read_kitti_calib_file should be read first")
            raise RuntimeError
        hfiller = np.expand_dims(np.ones(pts.shape[0]), axis=1)
        pts_hT = np.hstack([pts, hfiller]).T #(4, #pts)
        pts_lidar_T = inv(self.Tr_velo_to_cam) @ inv(self.R0_rect) @ pts_hT # (4, #pts)
        pts_lidar = pts_lidar_T.T # (#pts, 4)
        return pts_lidar[:, :3]

class KittiLabel:
    '''
    class storing KITTI 3d object detection label
        self.data ([KittiObj])    
    '''
    def __init__(self, label_path):
        self.path = label_path
        self.data = None

    def read_kitti_label_file(self, no_dontcare=True):
        '''
        read KITTI label file
        '''
        self.data = []
        with open(self.path, 'r') as f:
            str_list = f.readlines()
        str_list = [itm.rstrip() for itm in str_list if itm != '\n']
        for s in str_list:
            self.data.append(KittiObj(s))
        if no_dontcare:
            self.data = list(filter(lambda obj: obj.type != "DontCare", self.data))
        return self

class KittiObj():
    '''
    class storing a KITTI 3d object
    '''
    def __init__(self, s):
        self.type = None
        self.truncated = None
        self.occluded = None
        self.alpha = None
        self.bbox_l = None
        self.bbox_t = None
        self.bbox_r = None
        self.bbox_b = None
        self.h = None
        self.w = None
        self.l = None
        self.x = None
        self.y = None
        self.z = None
        self.ry = None
        self.score = None
        if len(s.split()) == 15: # data
            self.truncated, self.occluded, self.alpha,\
            self.bbox_l, self.bbox_t, self.bbox_r, self.bbox_b, \
            self.h, self.w, self.l, self.x, self.y, self.z, self.ry = \
            [float(itm) for itm in s.split()[1:]]
            self.type = s.split()[0]
        elif len(s.split()) == 16: # result
            self.truncated, self.occluded, self.alpha,\
            self.bbox_l, self.bbox_t, self.bbox_r, self.bbox_b, \
            self.h, self.w, self.l, self.x, self.y, self.z, self.ry, self.score = \
            [float(itm) for itm in s.split()[1:]]
            self.type = s.split()[0]
        else:
            raise NotImplementedError

    def __str__(self):
        if self.score is None:
            return "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(
                self.type, self.truncated, self.occluded, self.alpha,\
                self.bbox_l, self.bbox_t, self.bbox_r, self.bbox_b, \
                self.h, self.w, self.l, self.x, self.y, self.z, self.ry)
        else:
            return "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(
                self.type, self.truncated, self.occluded, self.alpha,\
                self.bbox_l, self.bbox_t, self.bbox_r, self.bbox_b, \
                self.h, self.w, self.l, self.x, self.y, self.z, self.ry, self.score)

    def get_bbox3dcorners(self):
        '''
        get the 8 corners of the bbox3d in camera frame.
        1.--.2
         |  |
         |  |
        4.--.3 (bottom)

        5.--.6
         |  |
         |  |
        8.--.7 (top)

        Camera Frame:
                   ^z
                   |
                y (x)----->x
        '''
        # lwh <-> xzy
        l, w, h = self.l, self.w, self.h
        x, z, y = self.x, self.z, self.y
        bottom = np.array([
            [-l/2, 0, w/2],
            [l/2, 0, w/2],
            [l/2, 0, -w/2],
            [-l/2, 0, -w/2],
        ])
        bottom = utils.apply_R(bottom, utils.roty(self.ry))
        bottom = utils.apply_tr(bottom, np.array([x, y, z]).reshape(-1, 3))
        top = utils.apply_tr(bottom, np.array([0, -h, 0]))
        return np.vstack([bottom, top])

class KittiData:
    '''
    class storing a frame of KITTI data
    '''
    def __init__(self, root_dir, idx):
        '''
        inputs:
            root_dir(str): kitti dataset dir
            idx(str %6d): data index e.g. "000000"
        '''
        self.calib_path = os.path.join(root_dir, "calib", idx+'.txt')
        self.image2_path = os.path.join(root_dir, "image_2", idx+'.png')
        self.label2_path = os.path.join(root_dir, "label_2", idx+'.txt')
        self.velodyne_path = os.path.join(root_dir, "velodyne", idx+'.bin')

    def read_data(self):
        '''
        read data
        returns:
            pc(np.array): [# of points, 4]
                point cloud in lidar frame.
                [x, y, z]
                      ^x
                      |
                y<----.z
        '''
        calib = KittiCalib(self.calib_path).read_kitti_calib_file()
        image = utils.read_image(self.image2_path)
        label = KittiLabel(self.label2_path)
        pc = utils.read_pc_from_bin(self.velodyne_path)
        return calib, image, label, pc

if __name__ == "__main__":
    label = KittiLabel("/usr/app/data/KITTI/dev/label_2/000009.txt").read_kitti_label_file()
    for obj in label.data:
        print(obj)
