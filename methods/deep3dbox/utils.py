'''
File Created: Friday, 10th May 2019 5:10:10 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import numpy as np
from scipy.optimize import lsq_linear
from det3.utils.utils import roty, istype
from det3.dataloader.kittidata import KittiObj

def recover_loc_by_geometry(K, ry, l, w, h, bbox2d, calib):
    '''
    recover location with the geometry constraints
    inputs:
        K (np.array) [3 x 4]: camera intrinx matrix
        ry (float) [3 x 3]: rotation around y axis in camera frame
        l w h (float): obj.l, obj.w, obj.h (obj is KittyObj)
        bbox2d (np.array) [4,] [xmin, xmax, ymin, ymax]
        calib (KittiCalib)
    returns:
        x y z (float): x, y, z
    '''
    f = K[0, 0]
    P13 = K[0, 2]
    P14 = K[0, 3]
    P23 = K[1, 2]
    P24 = K[1, 3]
    P34 = K[2, 3]
    R = roty(ry)
    r11 = R[0, 0]
    r13 = R[0, 2]
    r31 = R[2, 0]
    r33 = R[2, 2]
    cns_Fobj = np.array([
        [-l/2, 0, w/2],
        [l/2, 0, w/2],
        [l/2, 0, -w/2],
        [-l/2, 0, -w/2],
        [-l/2, -h, w/2],
        [l/2, -h, w/2],
        [l/2, -h, -w/2],
        [-l/2, -h, -w/2]
    ])
    bbox2d_x_min, bbox2d_x_max, bbox2d_y_min, bbox2d_y_max = bbox2d
    ious = []
    locs = []
    for idx, _ in np.ndenumerate(np.arange(8*8*8*8).reshape(8, 8, 8, 8)):
        if(idx[2] < 4):
            continue
        # x_0 -> x_min
        target = bbox2d_x_min
        xo_x = cns_Fobj[idx[0], 0]
        xo_y = cns_Fobj[idx[0], 1]
        xo_z = cns_Fobj[idx[0], 2]
        A1 = np.array([f, 0, P13 - target])
        b1 = np.array([target * P34
                       - (P13 - target)*(r31 * xo_x + r33* xo_z)
                       - P14 - f*(r11*xo_x + r13*xo_z)])
        # x_max
        target = bbox2d_x_max
        xo_x = cns_Fobj[idx[1], 0]
        xo_y = cns_Fobj[idx[1], 1]
        xo_z = cns_Fobj[idx[1], 2]
        A2 = np.array([f, 0, P13 - target])
        b2 = np.array([target * P34
                       - (P13 - target)*(r31 * xo_x + r33* xo_z)
                       - P14 - f*(r11*xo_x + r13*xo_z)])
        # y_min
        target = bbox2d_y_min
        xo_x = cns_Fobj[idx[2], 0]
        xo_y = cns_Fobj[idx[2], 1]
        xo_z = cns_Fobj[idx[2], 2]
        A3 = [0, f, P23 - target]
        b3 = [target * P34
              - (P23 - target)*(r31 * xo_x + r33* xo_z)
              - P24 - f*(xo_y)]
        # y_max
        target = bbox2d_y_max
        xo_x = cns_Fobj[idx[3], 0]
        xo_y = cns_Fobj[idx[3], 1]
        xo_z = cns_Fobj[idx[3], 2]
        A4 = [0, f, P23 - target]
        b4 = [target * P34
              - (P23 - target)*(r31 * xo_x + r33* xo_z)
              - P24 - f*(xo_y)]
        A = np.vstack([A1, A2, A3, A4])
        b = np.vstack([b1, b2, b3, b4]).flatten()
        res = lsq_linear(A, b)
        x, y, z = res.x
        obj = KittiObj()
        obj.x, obj.y, obj.z = x, y, z
        obj.h, obj.w, obj.l = h, w, l
        obj.ry = ry
        flag, iou = check_geometry_constriant(bbox2d, obj, calib)
        ious.append(iou)
        locs.append([x, y, z])
    ious = np.asarray(ious)
    locs = np.asarray(locs)
    x, y, z = locs[np.argmax(ious), :]
    return x, y, z

def check_geometry_constriant(bbox2d, bbox3d, calib):
    '''
    check bbox3d satisfy bbox2d or not, return ture if satisfy
    inputs:
        bbox2d (np.array) [4,] [xmin, xmax, ymin, ymax]
        bbox3d (KittiObj)
        calib (KittiCalib)
    returns:
        flag (bool) = true if iou > 0.69
        iou (float)
    '''
    cns_Fcam = bbox3d.get_bbox3dcorners()
    cns_Fimg = calib.leftcam2imgplane(cns_Fcam)
    x_min = min(cns_Fimg[:, 0])
    x_max = max(cns_Fimg[:, 0])
    y_min = min(cns_Fimg[:, 1])
    y_max = max(cns_Fimg[:, 1])
    area = (x_max - x_min + 1) * (y_max - y_min + 1)
    area_bbox2d = (bbox2d[1] - bbox2d[0] + 1) * (bbox2d[3] - bbox2d[2] + 1)

    xx1 = np.maximum(x_min, bbox2d[0])
    yy1 = np.maximum(y_min, bbox2d[2])
    xx2 = np.minimum(x_max, bbox2d[1])
    yy2 = np.minimum(y_max, bbox2d[3])
    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (area + area_bbox2d - inter)
    return ovr > 0.69, ovr

def filter_label_range(label, calib, x_range, y_range, z_range):
    '''
    filter label and get the objs in the xyz_range specified area.
    inputs:
        label (KittiLabel/CarlaLabel):
            Label read from txt file
        x_range (tuple): (x_min(float), x_max(float)) in Lidar Frame / IMU Frame
        y_range (tuple): (y_min(float), y_max(float)) in Lidar Frame / IMU Frame
        z_range (tuple): (z_min(float), z_max(float)) in Lidar Frame / IMU Frame
    return:
        label (KittiLabel/CarlaLabel)
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