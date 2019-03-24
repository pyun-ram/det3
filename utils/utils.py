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
    return np.array(Image.open(path, 'r'))

def read_pc_from_pcd(pcd_path):
    """Load PointCloud data from pcd file."""
    p = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
    return p

def read_pc_from_bin(bin_path):
    """Load PointCloud data from pcd file."""
    p = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return p

def rotx(t):
    ''' 3D Rotation about the x-axis.
    source: https://github.com/charlesq34/pointnet
    '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def roty(t):
    ''' Rotation about the y-axis.
    source: https://github.com/charlesq34/pointnet'''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def rotz(t):
    ''' Rotation about the z-axis.
    source: https://github.com/charlesq34/pointnet'''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

def apply_R(pts, R):
    '''
    apply Rotation Matrix on pts
    inputs:
        pts (np.array): [#pts, 3]
        R (np.array): [3,3]
            Rotation Matix
        Note: pts and R should match with each other.
    return:
        pts_R (np.array): [#pts, 3]
    '''
    return (R @ pts.T).T

def apply_tr(pts, tr):
    '''
    apply Translation Vector on pts
    inputs:
        pts (np.array): [#pts, 3]
        tr (np.array): [1, 3]
        Note: pts and tr should match with each other.
    return:
        pts_tr (np.array): [#pts, 3]
    '''
    return pts + tr