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
        self.calib_path = calib_path

    def read_kitti_calib_file(self):
        '''
        read KITTI calib file
        inputs:
            calib_path (str)
        outputs:
            calib (dict)
        '''
        calib = dict()
        with open(self.calib_path, 'r') as f:
            calib_list = f.readlines()
        calib_list = [itm.rstrip() for itm in calib_list if itm != '\n']
        for itm in calib_list:
            calib[itm.split(':')[0]] = itm.split(':')[1]
        for k, v in calib.items():
            calib[k] = [float(itm) for itm in v.split()]
        return calib

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
        label = KittiLabel(self.label2_path) #TODO
        pc = utils.read_pc_from_bin(self.velodyne_path)
        return calib, image, label, pc

if __name__ == "__main__":
    calib = KittiCalib("/usr/app/data/KITTI/dev/calib/000000.txt")
    calib.read_kitti_calib_file()
