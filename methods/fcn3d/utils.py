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

def voxelize_pc(pts, res, x_range, y_range, z_range):
    """
    Voxelize a point cloud into a grid
    inputs:
        pts (np.array): [#points, >=3]
            orders: [x,y,z]
        res (tuple): (dx(float), dy(float), dz(float))
            resolution of grid
        x_range (tuple): (x_min(float), x_max(float))
        y_range (tuple): (y_min(float), y_max(float))
        z_range (tuple): (z_min(float), z_max(float))
    return:
        grid (np.array): in zyx order of Lidar Frame
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    dx, dy, dz = res
    logic_x = np.logical_and(pts[:, 0] >= x_min, pts[:, 0] < x_max)
    logic_y = np.logical_and(pts[:, 1] >= y_min, pts[:, 1] < y_max)
    logic_z = np.logical_and(pts[:, 2] >= z_min, pts[:, 2] < z_max)
    pts_ = pts[:, :3][np.logical_and(logic_x, np.logical_and(logic_y, logic_z))].copy()
    pts_ = (pts_ - np.array([x_min, y_min, z_min])) / np.array([dx, dy, dz])
    pts_ = np.floor(pts_).astype(np.int32)
    grid = np.zeros(
        (np.floor((z_max - z_min)/dz).astype(np.int32),
         np.floor((y_max - y_min)/dy).astype(np.int32),
         np.floor((x_max - x_min)/dx).astype(np.int32))
        )
    grid[pts_[:, 2], pts_[:, 1], pts_[:, 0]] = 1
    return grid

def filter_label_cls(label, actuall_cls):
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

def lidar2grid(pts, res, x_range, y_range, z_range):
    '''
    get corresponding position of pts in the given grid which is defined by res, x, y, z
    inputs:
        pts (np.array): [#points, >=3]
            orders: [x,y,z]
        res (tuple): (dx(float), dy(float), dz(float))
            resolution of grid
        x_range (tuple): (x_min(float), x_max(float))
        y_range (tuple): (y_min(float), y_max(float))
        z_range (tuple): (z_min(float), z_max(float))
    returns:
        grid_zyx (np.array): [#points, 3]
            the corresponding position of pts in the given grid which is defined by res, x, y, z
            Note: The returned grid is with [z, y, x] order (z,y,x is in Lidar Frame)
    '''
    grid_x = np.floor((pts[:, 0:1] - x_range[0]) / res[0]).astype(np.int32)
    grid_y = np.floor((pts[:, 1:2] - y_range[0]) / res[1]).astype(np.int32)
    grid_z = np.floor((pts[:, 2:3] - z_range[0]) / res[2]).astype(np.int32)
    return np.hstack((grid_z, grid_y, grid_x))

def grid2lidar(gs, res, x_range, y_range, z_range, bias=None):
    '''
    recover the real position from the position in the grid
    inputs:
        gs (np.array): [#points, 3]
            orders: [z,y,x]
        res (tuple): (dx(float), dy(float), dz(float))
            resolution of grid
        x_range (tuple): (x_min(float), x_max(float))
        y_range (tuple): (y_min(float), y_max(float))
        z_range (tuple): (z_min(float), z_max(float))
        bias (None or np.array): [#points, 3]
            if bias is None,
                it will return the center coordinate corresponded to gs in Lidar Frame
            else
                it will adding each bias and return
    return:
        pts: [#points, 3]
            the corresponding position of voxel center recovered to the 3D space.
    '''
    if bias is None:
        bias = np.zeros((gs.shape[0], 3))
    gs_xyz = np.hstack([gs[:, 2:3], gs[:, 1:2], gs[:, 0:1]]) # order xyz
    pts_x = gs_xyz[:, 0:1] * res[0] + x_range[0] + res[0] / 2.0 + bias[:, 0:1]
    pts_y = gs_xyz[:, 1:2] * res[1] + y_range[0] + res[1] / 2.0 + bias[:, 1:2]
    pts_z = gs_xyz[:, 2:3] * res[2] + z_range[0] + res[2] / 2.0 + bias[:, 2:3]
    return np.hstack((pts_x, pts_y, pts_z))

def create_objectgrid(label, calib, res, x_range, y_range, z_range):
    '''
    create the object grid for training
    inputs:
        label (KittiLabel):
            filtered label
        calib (KittiCalib)
        res (tuple): (dx(float), dy(float), dz(float))
            resolution of grid
        x_range (tuple): (x_min(float), x_max(float))
        y_range (tuple): (y_min(float), y_max(float))
        z_range (tuple): (z_min(float), z_max(float))
    '''
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    dx, dy, dz = res
    centers_Fcam = [np.array([obj.x, obj.y-obj.h/2.0, obj.z]).reshape(-1, 3) for obj in label.data]
    centers_Fcam = np.vstack(centers_Fcam)
    centers_Flidar = calib.leftcam2lidar(centers_Fcam)
    centers_Fgrid = lidar2grid(centers_Flidar, res, x_range, y_range, z_range)
    grid = np.zeros(
        (1,
         np.floor((z_max - z_min)/dz).astype(np.int32),
         np.floor((y_max - y_min)/dy).astype(np.int32),
         np.floor((x_max - x_min)/dx).astype(np.int32))
        )
    for center_Fgrid in centers_Fgrid:
        z, y, x = center_Fgrid
        grid[0, z, y, x] = 1
    return grid

def create_regressgrid(label, calib, res, x_range, y_range, z_range):
    '''
    create the regression grid for training
    inputs:
        label (KittiLabel):
            filtered label
        calib (KittiCalib)
        res (tuple): (dx(float), dy(float), dz(float))
            resolution of grid
        x_range (tuple): (x_min(float), x_max(float))
        y_range (tuple): (y_min(float), y_max(float))
        z_range (tuple): (z_min(float), z_max(float))
    '''
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    dx, dy, dz = res
    centers_Fcam = [np.array([obj.x, obj.y-obj.h/2.0, obj.z]).reshape(-1, 3) for obj in label.data]
    centers_Fcam = np.vstack(centers_Fcam)
    centers_Flidar = calib.leftcam2lidar(centers_Fcam)
    centers_Fgrid = lidar2grid(centers_Flidar, res, x_range, y_range, z_range)
    cns_Fcam = [obj.get_bbox3dcorners() for obj in label.data]
    cns_Fcam = np.vstack(cns_Fcam)
    cns_Flidar = calib.leftcam2lidar(cns_Fcam).reshape(-1, 8, 3)
    rec_centers_Flidar = grid2lidar(centers_Fgrid, res, x_range, y_range, z_range)
    cns_Ftmp = [cns_Flidar_ - rec_centers_Flidar_ for cns_Flidar_, rec_centers_Flidar_ in zip(cns_Flidar, rec_centers_Flidar)]
    grid = np.zeros(
        (24, np.floor((z_max - z_min)/dz).astype(np.int32),
         np.floor((y_max - y_min)/dy).astype(np.int32),
         np.floor((x_max - x_min)/dx).astype(np.int32))
        )
    for cns_Ftmp_, center_Fgrid in zip(cns_Ftmp, centers_Fgrid):
        z, y, x = center_Fgrid
        grid[:, z, y, x] = cns_Ftmp_[:].flatten()
    return grid
