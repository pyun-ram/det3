'''
File Created: Friday, 29th March 2019 4:20:35 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import numpy as np

def filter_camera_angle(pts):
    """
    Filter camera angles (45 degrees) for KiTTI Datasets
    inputs:
        pts (np.array): [#points, >=3]
            orders: [x,y,z]
    return:
        pts in the camera angle (45degrees)
    """
    bool_in = np.logical_and((pts[:, 1] < pts[:, 0]), (-pts[:, 1] < pts[:, 0]))
    return pts[bool_in]

def pc_to_voxel(pts, res, x, y, z):
    """
    Voxelize a point cloud into a grid
    inputs:
        pts (np.array): [#points, >=3]
            orders: [x,y,z]
        res (tuple): (dx(float), dy(float), dz(float))
            resolution of grid
        x (tuple): (x_min(float), x_max(float))
        y (tuple): (y_min(float), y_max(float))
        z (tuple): (z_min(float), z_max(float))
    """
    logic_x = np.logical_and(pts[:, 0] >= x[0], pts[:, 0] < x[1])
    logic_y = np.logical_and(pts[:, 1] >= y[0], pts[:, 1] < y[1])
    logic_z = np.logical_and(pts[:, 2] >= z[0], pts[:, 2] < z[1])
    pts = pts[:, :3][np.logical_and(logic_x, np.logical_and(logic_y, logic_z))]
    pts = (pts - np.array([x[0], y[0], z[0]])) / np.array([res[0], res[1], res[2]])
    pts = np.ceil(pts).astype(np.int32)
    voxel = np.zeros(
        (np.ceil((x[1] - x[0])/res[0]).astype(np.int32),
         np.ceil((y[1] - y[0])/res[1]).astype(np.int32),
         np.ceil((z[1] - z[0])/res[2]).astype(np.int32))
        )
    voxel[pts[:, 0], pts[:, 1], pts[:, 2]] = 1
    return voxel

def filter_label(label, actuall_cls):
    '''
    filter label and get the objs in cls.
    inputs:
        label (KittiLabel):
            Kitti Label read from txt file
        cls (str):
            'Car', 'Pedestrian', 'Cyclist'
        actuall_cls (list):
            The actuall class correspond to cls.
            e.g. ['Car', 'Van'] for cls=='Car'
    '''
    label.data = list(filter(lambda obj: obj.type in actuall_cls, label.data))
    return label