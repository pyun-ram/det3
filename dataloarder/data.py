'''
File Created: Sunday, 17th March 2019 3:58:52 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import os
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
    '''
    def __init__(self, calib_path):
        self.path = calib_path
        self.data = None

    def read_kitti_calib_file(self):
        '''
        read KITTI calib file
        outputs:
            calib(dict):
                keys: 'P0', 'P1', 'P2', 'P3', 'R0_rect', 'Tr_velo_to_cam', 'Tr_imu_to_velo'
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
        return calib

class KittiLabel:
    '''
    class storing KITTI 3d object detection label
    '''
    def __init__(self, label_path):
        self.path = label_path
        self.data = None

    def read_kitti_label_file(self, no_dontcare=True):
        '''
        read KITTI label file
        outputs:
            label(list):
        '''
        self.data = []
        with open(self.path, 'r') as f:
            str_list = f.readlines()
        str_list = [itm.rstrip() for itm in str_list if itm != '\n']
        for s in str_list:
            self.data.append(KittiObj(s))
        if no_dontcare:
            self.data = list(filter(lambda obj: obj.type != "DontCare", self.data))
        return self.data

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
        '''
        calib = KittiCalib(self.calib_path).read_kitti_calib_file()
        image = utils.read_image(self.image2_path)
        label = KittiLabel(self.label2_path)
        pc = utils.read_pc_from_bin(self.velodyne_path)
        return calib, image, label, pc

if __name__ == "__main__":
    label = KittiLabel("/usr/app/data/KITTI/dev/label_2/000009.txt").read_kitti_label_file()
    for obj in label:
        print(obj)
