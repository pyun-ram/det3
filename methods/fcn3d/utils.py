'''
File Created: Friday, 29th March 2019 4:20:35 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import numpy as np
import sys
sys.path.append('../')
from det3.dataloarder.kittidata import KittiObj, KittiLabel
from det3.dataloarder.carladata import CarlaObj, CarlaLabel
from det3.utils.utils import istype, apply_R, apply_tr, rotz

def filter_camera_angle(pts):
    """
    Filter camera angles (45 degrees) for KiTTI/CARLA Datasets
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
        label (KittiLabel/CarlaLabel):
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
        label (KittiLabel/CarlaLabel):
            Kitti Label read from txt file
        x_range (tuple): (x_min(float), x_max(float)) in Lidar Frame / IMU Frame
        y_range (tuple): (y_min(float), y_max(float)) in Lidar Frame / IMU Frame
        z_range (tuple): (z_min(float), z_max(float)) in Lidar Frame / IMU Frame
    '''
    tmp = []
    for obj in label.data:
        if istype(label, "KittiLabel") and istype(calib, "KittiCalib"):
            cam_pt = np.array([[obj.x, obj.y-obj.h/2.0, obj.z]])
            lidar_pt = calib.leftcam2lidar(cam_pt)
        elif istype(label, "CarlaLabel") and istype(calib, "CarlaCalib"):
            lidar_pt = np.array([[obj.x, obj.y+obj.h/2.0, obj.z]]) # IMU Frame
        else:
            raise NotImplementedError
        if (x_range[0] <= lidar_pt[0, 0] <= x_range[1] and
            y_range[0] <= lidar_pt[0, 1] <= y_range[1] and
            z_range[0] <= lidar_pt[0, 2] <= z_range[1]):
            tmp.append(obj)
    label.data = tmp
    return label

def filter_label_pts(label, calib, pc, threshold=10):
    '''
    filter label and get the objs in the xyz_range specified area.
    inputs:
        label (KittiLabel/CarlaLabel)
        calib (KittiCalib/CarlaCalib)
        pc
            PC in LiDAR frame/IMU frame
    '''
    if istype(label, "CarlaLabel") and istype(calib, "CarlaCalib"):
        tmp = []
        for obj in label.data:
            #get pts from pc
            num_pts = obj.get_pts(pc, calib).shape[0]
            if num_pts > threshold:
                tmp.append(obj)
        label.data = tmp
        return label
    else:
        raise NotImplementedError

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
        label (KittiLabel/CarlaLabel):
            filtered label
        calib (KittiCalib/CarlaCalib)
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
    if label.isempty():
        return np.zeros(
            (1,
             np.floor((z_max - z_min)/dz).astype(np.int32),
             np.floor((y_max - y_min)/dy).astype(np.int32),
             np.floor((x_max - x_min)/dx).astype(np.int32))
            )
    if istype(label, "KittiLabel") and istype(calib, "KittiCalib"):
        centers_Fcam = [np.array([obj.x, obj.y-obj.h/2.0, obj.z]).reshape(-1, 3) for obj in label.data]
        centers_Fcam = np.vstack(centers_Fcam)
        centers_Flidar = calib.leftcam2lidar(centers_Fcam)
        centers_Fgrid = lidar2grid(centers_Flidar, res, x_range, y_range, z_range)
    elif istype(label, "CarlaLabel") and istype(calib, "CarlaCalib"):
        centers_FIMU = [np.array([obj.x, obj.y+obj.h/2.0, obj.z]).reshape(-1, 3) for obj in label.data]
        centers_FIMU = np.vstack(centers_FIMU)
        centers_Fgrid = lidar2grid(centers_FIMU, res, x_range, y_range, z_range)
    else:
        raise NotImplementedError
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
        label (KittiLabel/CarlaLabel):
            filtered label
        calib (KittiCalib/CarlaCalib)
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
    if label.isempty():
        return np.zeros(
            (24, np.floor((z_max - z_min)/dz).astype(np.int32),
             np.floor((y_max - y_min)/dy).astype(np.int32),
             np.floor((x_max - x_min)/dx).astype(np.int32))
            )
    if istype(label, "KittiLabel") and istype(calib, "KittiCalib"):
        centers_Fcam = [np.array([obj.x, obj.y-obj.h/2.0, obj.z]).reshape(-1, 3) for obj in label.data]
        centers_Fcam = np.vstack(centers_Fcam)
        centers_Flidar = calib.leftcam2lidar(centers_Fcam)
        centers_Fgrid = lidar2grid(centers_Flidar, res, x_range, y_range, z_range)
        cns_Fcam = [obj.get_bbox3dcorners() for obj in label.data]
        cns_Fcam = np.vstack(cns_Fcam)
        cns_Flidar = calib.leftcam2lidar(cns_Fcam).reshape(-1, 8, 3)
        rec_centers_Flidar = grid2lidar(centers_Fgrid, res, x_range, y_range, z_range)
        cns_Ftmp = [cns_Flidar_ - rec_centers_Flidar_ for cns_Flidar_, rec_centers_Flidar_ in zip(cns_Flidar, rec_centers_Flidar)]
    elif istype(label, "CarlaLabel") and istype(calib, "CarlaCalib"):
        centers_FIMU = [np.array([obj.x, obj.y+obj.h/2.0, obj.z]).reshape(-1, 3) for obj in label.data]
        centers_FIMU = np.vstack(centers_FIMU)
        centers_Fgrid = lidar2grid(centers_FIMU, res, x_range, y_range, z_range)
        cns_FIMU = [obj.get_bbox3dcorners() for obj in label.data]
        cns_FIMU = np.vstack(cns_FIMU)
        cns_FIMU = cns_FIMU.reshape(-1, 8, 3)
        rec_centers_FIMU = grid2lidar(centers_Fgrid, res, x_range, y_range, z_range)
        cns_Ftmp = [cns_FIMU_ - rec_centers_FIMU_ for cns_FIMU_, rec_centers_FIMU_ in zip(cns_FIMU, rec_centers_FIMU)]
    grid = np.zeros(
        (24, np.floor((z_max - z_min)/dz).astype(np.int32),
         np.floor((y_max - y_min)/dy).astype(np.int32),
         np.floor((x_max - x_min)/dx).astype(np.int32))
        )
    for cns_Ftmp_, center_Fgrid in zip(cns_Ftmp, centers_Fgrid):
        z, y, x = center_Fgrid
        grid[:, z, y, x] = cns_Ftmp_[:].flatten()
    return grid

def cnsFcam2d_to_bboxes2d(cns_Fcam2d):
    '''
    convert corners in left camera plane to 2D bounding boxes
    inputs:
        cnsFcam2d (np.array): [#bboxes, 8, 2]
    return:
        bboxes2d (np.array): [#bboxes, 4]
            [minx, maxx, miny, maxy]
    '''
    minx_Fcam2d = np.min(cns_Fcam2d[:, :, 0:1], axis=1).astype(np.int)
    maxx_Fcam2d = np.max(cns_Fcam2d[:, :, 0:1], axis=1).astype(np.int)
    miny_Fcam2d = np.min(cns_Fcam2d[:, :, 1:2], axis=1).astype(np.int)
    maxy_Fcam2d = np.max(cns_Fcam2d[:, :, 1:2], axis=1).astype(np.int)
    boxes2d = np.hstack([minx_Fcam2d, maxx_Fcam2d, miny_Fcam2d, maxy_Fcam2d])
    return boxes2d

# TODO: oriented nms on bev img
# https://github.com/aaronfriedman6/MV3D_VoxelNet/blob/5184cb327a69da4056784d3070a90592a536a9c9/MV3D/src/tracklets/evaluate_tracklets.py#L27
def nms(boxes, scores, threshold):
    '''
    Non-Maximum Surpression (NMS).
    inputs:
        boxes (np.array): [# boxes, 4]
            2D boxes [minx, maxx, miny, maxy]
        scores (np.array): (# boxes, )
    return:
        idx (list)
    reference:https://www.cnblogs.com/makefile/p/nms.html
    '''
    x1 = boxes[:, 0]
    x2 = boxes[:, 1]
    y1 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
    return keep

def parse_grid_to_label(obj_grid, reg_grid, score_threshold, nms_threshold, calib, cls, res, x_range, y_range, z_range):
    '''
    get the estimation from obj_grid and reg_grid
    inputs:
        obj_grid (np.array): [1, z_dim, y_dim, x_dim] (LiDAR Frame)
        reg_grid (np.array): [24, z_dim, y_dim, x_dim] (LiDAR Frame)
        threshold (float):
            the posterior probability >= threshold in the obj_grid will be returned.
        nms_threshold (float):
            the threshold for NMS IoU selection
        calib (KittiCalib/CarlaCalib)
        cls (str)
        res (tuple): (dx(float), dy(float), dz(float))
            resolution of grid
        x_range (tuple): (x_min(float), x_max(float))
        y_range (tuple): (y_min(float), y_max(float))
        z_range (tuple): (z_min(float), z_max(float))
    return:
        res: (KittiLabel/CarlaLabel)
    '''
    num_of_objs = np.vstack(np.where(obj_grid >= score_threshold)).T.shape[0]
    # get scores
    scores = obj_grid[obj_grid >= score_threshold]
    if istype(calib, "KittiCalib"):
        if num_of_objs == 0:
            label = KittiLabel()
            label.data = []
            return label
        # get cns_Fcam
        centers_Fgrid = np.vstack(np.where(obj_grid >= score_threshold)).T[:, 1:]
        centers_Flidar = grid2lidar(centers_Fgrid, res, x_range, y_range, z_range, bias=None)
        bias_Ftmp = [reg_grid[:, center_Fgrid[0], center_Fgrid[1], center_Fgrid[2]].reshape(8, 3) for center_Fgrid in centers_Fgrid]
        cns_Flidar = [_bias_Ftmp + center_Flidar for _bias_Ftmp, center_Flidar in zip(bias_Ftmp, centers_Flidar)]
        cns_Flidar = np.vstack(cns_Flidar)
        cns_Fcam = calib.lidar2leftcam(cns_Flidar)
        # get cns_Fcam2d
        cns_Fcam2d = calib.leftcam2imgplane(cns_Fcam)
        cns_Fcam = cns_Fcam.reshape(-1, 8, 3)
        cns_Fcam2d = cns_Fcam2d.reshape(-1, 8, 2)
        # get boxes2d
        boxes2d = cnsFcam2d_to_bboxes2d(cns_Fcam2d)
        # nms
        idx = nms(boxes2d, scores, nms_threshold)
        label = KittiLabel()
        label.data = []
        for _cns_Fcam, score in zip(cns_Fcam[idx], scores[idx]):
            label.data.append(KittiObj().from_corners(calib, _cns_Fcam, cls, score))
        return label
    elif istype(calib, "CarlaCalib"):
        if num_of_objs == 0:
            label = CarlaLabel()
            label.data = []
            return label
        # get cns_Fcam
        centers_Fgrid = np.vstack(np.where(obj_grid >= score_threshold)).T[:, 1:]
        centers_FIMU = grid2lidar(centers_Fgrid, res, x_range, y_range, z_range, bias=None)
        bias_Ftmp = [reg_grid[:, center_Fgrid[0], center_Fgrid[1], center_Fgrid[2]].reshape(8, 3) for center_Fgrid in centers_Fgrid]
        cns_FIMU = [_bias_Ftmp + center_FIMU for _bias_Ftmp, center_FIMU in zip(bias_Ftmp, centers_FIMU)]
        cns_FIMU = np.vstack(cns_FIMU)
        cns_Fcam = calib.imu2cam(cns_FIMU)
        # get cns_Fcam2d
        cns_Fcam2d = calib.cam2imgplane(cns_Fcam)
        cns_Fcam = cns_Fcam.reshape(-1, 8, 3)
        cns_Fcam2d = cns_Fcam2d.reshape(-1, 8, 2)
        # get boxes2d
        boxes2d = cnsFcam2d_to_bboxes2d(cns_Fcam2d)
        # nms
        idx = nms(boxes2d, scores, nms_threshold)
        label = CarlaLabel()
        label.data = []
        for _cns_Fcam, score in zip(cns_Fcam[idx], scores[idx]):
            label.data.append(CarlaObj().from_corners(calib, _cns_Fcam, cls, score))
        return label
    else:
        raise NotImplementedError
