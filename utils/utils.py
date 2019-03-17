'''
File Created: Sunday, 17th March 2019 9:41:35 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import numpy as np
from PIL import Image

def read_image(path):
    '''
    read image
    inputs:
        path(str): image path
    returns:
        img(np.array): [w,h,c]
    '''
    return np.array(Image.open(path, '3'))

def read_pc_from_pcd(pcd_path):
    """Load PointCloud data from pcd file."""
    p = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
    return p

def read_pc_from_bin(bin_path):
    """Load PointCloud data from pcd file."""
    p = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return p
