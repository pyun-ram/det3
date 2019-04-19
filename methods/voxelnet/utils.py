'''
File Created: Friday, 19th April 2019 2:42:58 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import numpy as np
import sys
sys.path.append("../")

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

def voxelize_pc(pts, res, x_range, y_range, z_range, num_pts_in_vox):
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
        num_pts_in_vox (int): 
            # of points in a voxel        
    return:
        grid (np.array): in zyx order of Lidar Frame
    reference: 
        https://github.com/qianguih/voxelnet
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    dx, dy, dz = res

    logic_x = np.logical_and(pts[:, 0] >= x_min, pts[:, 0] < x_max)
    logic_y = np.logical_and(pts[:, 1] >= y_min, pts[:, 1] < y_max)
    logic_z = np.logical_and(pts[:, 2] >= z_min, pts[:, 2] < z_max)
    pts_Flidar = pts[:, :4][np.logical_and(logic_x, np.logical_and(logic_y, logic_z))].copy()
    pts_Fgrid = np.floor((pts_Flidar[:, :3] - np.array([x_min, y_min, z_min],  dtype=np.float32)) 
                         / np.array([dx, dy, dz], dtype=np.float32))
    pts_Fgrid = pts_Fgrid[:, ::-1] # x y z -> z y x in LiDAR frame

    coordinate_buffer = np.unique(pts_Fgrid, axis=0)
    K = len(coordinate_buffer)
    T = num_pts_in_vox
    number_buffer = np.zeros(shape=(K), dtype=np.int64)
    feature_buffer = np.zeros(shape=(K, T, 7), dtype=np.float32)

    # build a reverse index for coordinate buffer
    index_buffer = {}
    for i in range(K):
        index_buffer[tuple(coordinate_buffer[i])] = i

    for pt_Fgrid, pt_Flidar in zip(pts_Fgrid, pts_Flidar):
        index = index_buffer[tuple(pt_Fgrid)]
        number = number_buffer[index]
        if number < T:
            feature_buffer[index, number, :4] = pt_Flidar
            number_buffer[index] += 1
    feature_buffer[:, :, -3:] = (feature_buffer[:, :, :3] -
                                 feature_buffer[:, :, :3].sum(axis=1, keepdims=True)
                                 / number_buffer.reshape(K, 1, 1))
    mask = np.not_equal(feature_buffer[:, :, :3].max(axis=2, keepdims=True), 0)
    mask = np.tile(mask, [1, 1, 3])
    feature_buffer[:, :, -3:] = mask.astype(np.float32) * feature_buffer[:, :, -3:]
    voxel_dict = {'feature_buffer': feature_buffer,
                  'coordinate_buffer': coordinate_buffer,
                  'number_buffer': number_buffer}
    return voxel_dict

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

def filter_label_range(label, calib, x_range, y_range, z_range):
    '''
    filter label and get the objs in the xyz_range specified area.
    inputs:
        label (KittiLabel):
            Kitti Label read from txt file
        x_range (tuple): (x_min(float), x_max(float)) in Lidar Frame
        y_range (tuple): (y_min(float), y_max(float)) in Lidar Frame
        z_range (tuple): (z_min(float), z_max(float)) in Lidar Frame
    '''
    tmp = []
    for obj in label.data:
        cam_pt = np.array([[obj.x, obj.y-obj.h/2.0, obj.z]])
        lidar_pt = calib.leftcam2lidar(cam_pt)
        if (x_range[0] <= lidar_pt[0, 0] <= x_range[1] and
                y_range[0] <= lidar_pt[0, 1] <= y_range[1] and
                z_range[0] <= lidar_pt[0, 2] <= z_range[1]):
            tmp.append(obj)
    label.data = tmp
    return label

def create_anchors(x_range, y_range, target_shape, anchor_z, anchor_size):
    '''
    create anchors for regression map
    inputs:
        x_range (tuple): (x_min(float), x_max(float)) in Lidar Frame
        y_range (tuple): (y_min(float), y_max(float)) in Lidar Frame
        target_shape (tuple): (y_size (int), x_size(int))
            final output shape of VoxelNet in y and x dimension
        anchor_z (float):
            z value of anchors in LiDAR Frame
        anchor_size (tuple): (l(float), w(float), h(float))
            lwh <-> xzy (camera) <->yxz(lidar)
    return:
        anchors (np.array): [y_size, x_size, 2, 7] (cx, cy, cz, h, w, l, r) in lidar frame
    reference:
        https://github.com/qianguih/voxelnet
    '''
    x_min, x_max = x_range
    y_min, y_max = y_range
    y_size, x_size = target_shape
    l_, w_, h_ = anchor_size

    x = np.linspace(x_min, x_max, x_size)
    y = np.linspace(y_min, y_max, y_size)
    cx, cy = np.meshgrid(x, y)
    cx = np.tile(cx[..., np.newaxis], 2)
    cy = np.tile(cy[..., np.newaxis], 2)
    cz = np.ones_like(cx) * anchor_z
    w = np.ones_like(cx) *  w_
    l = np.ones_like(cx) * l_
    h = np.ones_like(cx) * h_
    r = np.ones_like(cx)
    r[..., 0] = 0  # 0
    r[..., 1] = 90 / 180.0 * np.pi  # 90
    # 7*(w,l,2) -> (w, l, 2, 7)
    anchors = np.stack([cx, cy, cz, h, w, l, r], axis=-1)

    return anchors