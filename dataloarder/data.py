'''
File Created: Sunday, 17th March 2019 3:58:52 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import os
import det3.utils.utils as utils

# KITTI
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
        calib = KittiCalib(self.calib_path) #TODO
        image = utils.read_image(self.image2_path)
        label = KittiLabel(self.label2_path) #TODO
        pc = utils.read_pc_from_bin(self.velodyne_path)
        return calib, image, label, pc
