'''
File Created: Friday, 19th April 2019 2:42:58 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import numpy as np
import sys
sys.path.append("../")
from det3.methods.voxelnet.box_overlaps import bbox_overlaps
from det3.utils.utils import istype, rotz, apply_R, apply_tr
from det3.dataloarder.kittidata import KittiLabel, KittiObj
from det3.dataloarder.carladata import CarlaLabel, CarlaObj

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
    pts_Fgrid = np.floor((pts_Flidar[:, :3] - np.array([x_min, y_min, z_min], dtype=np.float32))
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
        label (KittiLabel/CarlaLabel):
            Label read from txt file
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
            Label read from txt file
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

def filter_label_pts(label, pc, calib, threshold_pts=10):
    '''
    filter label and get the objs with #pts >= threshold_pts.
    inputs:
        label (KittiLabel/CarlaLabel):
            Label read from txt file
        pc (np.array) [#pts, 3]
            point cloud in LiDAR Frame
        calib (KittiCalib/CarlaLabel)
        threshold_pts (int)
    '''
    tmp = []
    for obj in label.data:
        if obj.get_pts(pc, calib).shape[0] >= threshold_pts:
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

def create_rpn_target(label, calib, target_shape, anchors, threshold_pos_iou, threshold_neg_iou, anchor_size):
    '''
    create target for regression and classfication
    inputs:
        label (KittiLabel/CarlaLabel)
        calib (KittiCalib/CarlaCalib)
        target_shape (tuple): (y_size (int), x_size(int))
            final output shape of VoxelNet in y and x dimension
        anchors (np.array): [y_size, x_size, 2, 7] (cx, cy, cz, h, w, l, r) in lidar frame
        threshold_pos_iou (float):
            anchor iou > threshold_pos_iou is counted as pos anchor
        threshold_neg_iou (float)
            anchor iou < threshold_neg_iou is counted as neg anchor
        anchor_size (tuple): (l(float), w(float), h(float))
            lwh <-> xzy (camera) <->yxz(lidar)            
    return:
        pos_equal_one, neg_equal_one, targets
    references:
        https://github.com/qianguih/voxelnet
    '''
    _, _, anchor_h = anchor_size
    targets = np.zeros((*target_shape, 14))
    pos_equal_one = np.zeros((*target_shape, 2))
    neg_equal_one = np.zeros((*target_shape, 2))
    if label.isempty():
        neg_equal_one = np.ones((*target_shape, 2))
        return pos_equal_one, neg_equal_one, targets
    gt_boxes3d = label_to_gt_box3d(label, calib)
    anchors_reshaped = anchors.reshape(-1, 7)
    anchors_d = np.sqrt(anchors_reshaped[:, 4]**2 + anchors_reshaped[:, 5]**2)
    anchors_standup_2d = anchor_to_standup_box2d(anchors_reshaped[:, [0, 1, 4, 5]])

    if istype(label, "KittiLabel") and istype(calib, "KittiCalib"):
        gtcns_Fcam = [obj.get_bbox3dcorners() for obj in label.data]
        gtcns_Fcam = np.vstack(gtcns_Fcam)
        cns_Flidar = calib.leftcam2lidar(gtcns_Fcam).reshape(-1, 8, 3)
        cns_Flidar_2d = cns_Flidar[:, :4, :2]
        gt_standup_2d = corner_to_standup_box2d(cns_Flidar_2d)
    elif istype(label, "CarlaLabel") and istype(calib, "CarlaCalib"):
        cns_FIMU = [obj.get_bbox3dcorners() for obj in label.data]
        cns_FIMU = np.vstack(cns_FIMU)
        cns_FIMU = cns_FIMU.reshape(-1, 8, 3)
        cns_FIMU_2d = cns_FIMU[:, :4, :2]
        gt_standup_2d = corner_to_standup_box2d(cns_FIMU_2d)
    else:
        raise NotImplementedError
    iou = bbox_overlaps(
        np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
        np.ascontiguousarray(gt_standup_2d).astype(np.float32),
    )

    # find anchor with highest iou(iou should also > 0)
    id_highest = np.argmax(iou.T, axis=1)
    id_highest_gt = np.arange(iou.T.shape[0])
    mask = iou.T[id_highest_gt, id_highest] > 0
    id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

    # find anchor iou > cfg.XXX_POS_IOU
    id_pos, id_pos_gt = np.where(iou > threshold_pos_iou)
    # find anchor iou < cfg.XXX_NEG_IOU
    id_neg = np.where(np.sum(iou < threshold_neg_iou,
                             axis=1) == iou.shape[1])[0]

    id_pos = np.concatenate([id_pos, id_highest])
    id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])

    # TODO: uniquify the array in a more scientific way
    id_pos, index = np.unique(id_pos, return_index=True)
    id_pos_gt = id_pos_gt[index]
    id_neg.sort()

    # cal the target and set the equal one
    index_x, index_y, index_z = np.unravel_index(
        id_pos, (*target_shape, 2))
    pos_equal_one[index_x, index_y, index_z] = 1

    # ATTENTION: index_z should be np.array
    targets[index_x, index_y, np.array(index_z) * 7] = (
        gt_boxes3d[id_pos_gt, 0] - anchors_reshaped[id_pos, 0]) / anchors_d[id_pos]
    targets[index_x, index_y, np.array(index_z) * 7 + 1] = (
        gt_boxes3d[id_pos_gt, 1] - anchors_reshaped[id_pos, 1]) / anchors_d[id_pos]
    targets[index_x, index_y, np.array(index_z) * 7 + 2] = (
        gt_boxes3d[id_pos_gt, 2] - anchors_reshaped[id_pos, 2]) / anchor_h
    targets[index_x, index_y, np.array(index_z) * 7 + 3] = np.log(
        gt_boxes3d[id_pos_gt, 3] / anchors_reshaped[id_pos, 3])
    targets[index_x, index_y, np.array(index_z) * 7 + 4] = np.log(
        gt_boxes3d[id_pos_gt, 4] / anchors_reshaped[id_pos, 4])
    targets[index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
        gt_boxes3d[id_pos_gt, 5] / anchors_reshaped[id_pos, 5])
    targets[index_x, index_y, np.array(index_z) * 7 + 6] = (
        gt_boxes3d[id_pos_gt, 6] - anchors_reshaped[id_pos, 6])

    index_x, index_y, index_z = np.unravel_index(
        id_neg, (*target_shape, 2))
    neg_equal_one[index_x, index_y, index_z] = 1
    # to avoid a box be pos/neg in the same time
    index_x, index_y, index_z = np.unravel_index(
        id_highest, (*target_shape, 2))
    neg_equal_one[index_x, index_y, index_z] = 0

    return pos_equal_one, neg_equal_one, targets

def parse_grid_to_label(obj_map, reg_map, anchors, anchor_size, cls, calib, threshold_score, threshold_nms):
    '''
        parse the regression map to labels
        inputs:
            obj_map (np.array) [target_shape[0], target_shape[1], 2]
            reg_map (np.array) [target_shape[0], target_shape[1], 14]
            anchors (np.array) [target_shape[0], target_shape[1], 2, 7]
            anchor_size (tuple): (l(float), w(float), h(float))
                lwh <-> xzy (camera) <->yxz(lidar)
            cls (str)
            calib (KittiCalib/CarlaCalib)
            threshold_score (float):
                result with score >= threshold_score will be counted
            threshold_nms (float):
                result with iou <= threshold_iou will be counted
        returns:
            label
        reference:
            https://github.com/qianguih/voxelnet
    '''
    obj_map = obj_map.transpose(1, 2, 0)
    reg_map = reg_map.transpose(1, 2, 0)
    _, _, anchor_h = anchor_size
    anchors_reshaped = anchors.reshape(-1, 7)
    deltas = reg_map.reshape(-1, 7)
    anchors_d = np.sqrt(anchors_reshaped[:, 4]**2 + anchors_reshaped[:, 5]**2)
    boxes3d = np.zeros_like(deltas)
    boxes3d[..., [0, 1]] = deltas[..., [0, 1]] * \
        anchors_d[:, np.newaxis] + anchors_reshaped[..., [0, 1]]
    boxes3d[..., [2]] = deltas[..., [2]] * \
        anchor_h + anchors_reshaped[..., [2]]
    boxes3d[..., [3, 4, 5]] = np.exp(
        deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
    boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

    boxes2d = boxes3d[:, [0, 1, 4, 5, 6]]
    probs = obj_map.reshape(-1)

    ind = np.where(probs[:] >= threshold_score)[0]
    if ind.shape[0] == 0 and istype(calib, "KittiCalib"):
        label = KittiLabel()
        label.data = []
        return label
    elif ind.shape[0] == 0 and istype(calib, "CarlaCalib"):
        label = CarlaLabel()
        label.data = []
        return label
    tmp_boxes3d = boxes3d[ind, ...]
    tmp_boxes2d = boxes2d[ind, ...]
    tmp_scores = probs[ind]
    boxes2d = center_to_standup_box2d(tmp_boxes2d)
    boxes2d = boxes2d[:, [0, 2, 1, 3]]
    # ind = nms(boxes2d, tmp_scores, threshold_nms)
    ind = nms_general(tmp_boxes3d, tmp_scores, threshold_nms,mode='2d-rot')
    tmp_boxes3d = tmp_boxes3d[ind, ...]
    tmp_scores = tmp_scores[ind, ...]
    if istype(calib, "KittiCalib"):
        label = KittiLabel()
        label.data = []
        for box3d, score in zip(tmp_boxes3d, tmp_scores):
            obj = KittiObj()
            x, y, z, h, w, l, ry = box3d
            btmcenter_Flidar = np.array([x, y, z]).reshape(1, -1)
            btmcenter_Fcam = calib.lidar2leftcam(btmcenter_Flidar)
            obj.x, obj.y, obj.z = btmcenter_Fcam.flatten()
            obj.h, obj.w, obj.l, obj.ry = h, w, l, ry
            cns_Fcam = obj.get_bbox3dcorners()
            obj.from_corners(calib, cns_Fcam, cls, score)
            label.data.append(obj)
    elif istype(calib, "CarlaCalib"):
        label = CarlaLabel()
        label.data = []
        for box3d, score in zip(tmp_boxes3d, tmp_scores):
            obj = CarlaObj()
            x, y, z, h, w, l, ry = box3d
            obj.x, obj.y, obj.z = x, y, z
            obj.h, obj.w, obj.l, obj.ry = h, w, l, ry
            cns_FIMU = obj.get_bbox3dcorners()
            cns_Fcam = calib.imu2cam(cns_FIMU)
            obj.from_corners(calib, cns_Fcam, cls, score)
            label.data.append(obj)
    else:
        raise NotImplementedError
    return label

def label_to_gt_box3d(label, calib):
    '''
    convert label into numpy array for further parallel operation
    inputs:
        label (KittiLabel/CarlaLabel)
        calib (KittiCalib/CarlaCalib)
    returns:
        label_np (np.array) [#obj, 7]
    '''
    boxes3d = []
    if label.isempty():
        return None
    for obj in label.data:
        h = obj.h
        w = obj.w
        l = obj.l
        ry = obj.ry
        if istype(label, "KittiLabel") and istype(calib, "KittiCalib"):
            btmcenter_Flidar = calib.leftcam2lidar(np.array([obj.x, obj.y, obj.z]).reshape(1, -1))
            x, y, z = btmcenter_Flidar.reshape(-1)
            box3d = [x, y, z, h, w, l, ry]
        elif istype(label, "CarlaLabel") and istype(calib, "CarlaCalib"):
            x, y, z = obj.x, obj.y, obj.z
            box3d = [x, y, z, h, w, l, ry]
        else:
            raise NotImplementedError
        boxes3d.append(np.array(box3d).reshape(-1, 7))
    return np.vstack(boxes3d).reshape(-1, 7)

def anchor_to_standup_box2d(anchors):
    '''
    convert anchor represented by [x,y,w,l] -> [x1, y1, x2, y2]
    inputs:
        anchors (np.array): (#anchors, 4) [x,y,w,l]
    returns:
        anchors (np.array): (#anchors, 4) [x1, y1, x2, y2]
    reference:
        https://github.com/qianguih/voxelnet
    '''
    anchor_standup = np.zeros_like(anchors)
    # r == 0
    anchor_standup[::2, 0] = anchors[::2, 0] - anchors[::2, 3] / 2
    anchor_standup[::2, 1] = anchors[::2, 1] - anchors[::2, 2] / 2
    anchor_standup[::2, 2] = anchors[::2, 0] + anchors[::2, 3] / 2
    anchor_standup[::2, 3] = anchors[::2, 1] + anchors[::2, 2] / 2
    # r == pi/2
    anchor_standup[1::2, 0] = anchors[1::2, 0] - anchors[1::2, 2] / 2
    anchor_standup[1::2, 1] = anchors[1::2, 1] - anchors[1::2, 3] / 2
    anchor_standup[1::2, 2] = anchors[1::2, 0] + anchors[1::2, 2] / 2
    anchor_standup[1::2, 3] = anchors[1::2, 1] + anchors[1::2, 3] / 2

    return anchor_standup
def center_to_standup_box2d(boxes_center):
    '''
    convert box2d represented by x,y,w,l,ry into standup_box2d [x1, y1, x2, y2]
    inputs:
        boxes_corner (np.array) [#boxes, 5]
            in Lidar Frame
    returns:
        standup_boxes2d (np.array) [#boxes, 4] [x1, y1, x2, y2]
    reference:
        https://github.com/qianguih/voxelnet
    '''
    # (N, 5) -> (N, 4) x1, y1, x2, y2
    num_boxes = boxes_center.shape[0]
    boxes_corner = []
    for i in range(num_boxes):
        x, y, w, l, ry = boxes_center[i, :]
        box_corner = np.array([
            [-l/2, w/2,  0 ],
            [-l/2, -w/2, 0],
            [ l/2, -w/2, 0],
            [ l/2, w/2,  0]
        ])
        box_corner = apply_R(box_corner, rotz(ry))
        box_corner = apply_tr(box_corner, np.array([x, y, 0]))
        boxes_corner.append(box_corner[:, :2])
    boxes_corner = np.vstack(boxes_corner).reshape(-1, 4, 2)
    standup_boxes2d = corner_to_standup_box2d(boxes_corner)
    return standup_boxes2d

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

def nms_general(boxes, scores, threshold, mode='2d-rot'):
    '''
    Non-Maximum Surpression (NMS).
    inputs:
        boxes (np.array) [# boxes, 7]
            3D boxes [x, y, z, l, w, h, ry]
        scores (np.array) [# boxes, ]
        threshold (float): keep the boxes with the IoU <= thresold
        mode (str): '2d', '2d-rot', '3d-rot'
    return:
        idx (list)
    '''
    # sort in descending order of scores
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i) # keep the one with highest score
        ovr = compute_iou(boxes[i:i+1], boxes[order[1:]], mode=mode) # compute iou with others
        inds = np.where(ovr <= threshold)[0] # keep the boxes with the IoU <= thresold
        order = order[inds + 1]
    return keep

def compute_iou(box, others, mode='2d-rot'):
    '''
    compute IoU between box and others.
    inputs:
        box (np.array) [1, 7]
            3D boxes [x, y, z, l, w, h, ry]
        others (np.array) [#boxes, 7]
            3D boxes [x, y, z, l, w, h, ry]
        mode (str): '2d-rot', '3d-rot'
    return:
        ious(np.array) [#boxes,]
    '''
    if mode == '2d-rot':
        # compute box area
        box_area = compute_area(box)
        # compute others area
        others_area = compute_area(others)
        # compute intersection
        inter = compute_intersec(box, others, mode)
        # compute iou
        iou = inter / (box_area + others_area - inter)
    elif mode == '3d-rot':
        pass
    else:
        raise NotImplementedError
    return iou

def compute_area(boxes):
    '''
    compute area of boxes.
    inputs:
        boxes (np.array) [#boxes, 7]
    return:
        areas (np.array) [#boxes, ]
    '''
    return boxes[:, 3] * boxes[:, 4]

def compute_intersec(box, others, mode):
    '''
    compute intersect between box and others.
    inputs:
        box (np.array) [1, 7]
            3D boxes [x, y, z, l, w, h, ry]
        others (np.array) [#boxes, 7]
            3D boxes [x, y, z, l, w, h, ry]
        mode (str): '2d-rot', '3d-rot'
    return:
        inter(np.array) [#boxes,]
    '''
    # from shapely.geometry import Polygon
    num_all = others.shape[0]+1 # total number of all boxes (box + other)
    all_boxes = np.vstack([box, others]) # num_all, 7 (x, y, z, l, w, h, ry)
    all_plg2d = boxes3d2polygon(all_boxes) # convert boxes3d into 2d polygons for intersection computing
    if mode == '2d-rot':
        inter = np.zeros(others.shape[0])
        for i, plg in enumerate(all_plg2d[1:]):
            inter[i] = all_plg2d[0].intersection(plg).area
    return inter

def boxes3d2polygon(boxes):
    '''
    convert boxes defined by [x, y, z, l, w, h, ry] into polygons
    inputs:
        boxes: (#boxes, 7) [x, y, z, l, w, h, ry]
    outputs:
        polygons (list) [#boxes]
    '''
    from shapely.geometry import Polygon
    num_all = boxes.shape[0]
    all_cns = np.zeros([num_all, 8, 3])
    all_x = boxes[:, 0:1]
    all_y = boxes[:, 1:2]
    all_z = boxes[:, 2:3]
    all_l = boxes[:, 3:4]
    all_w = boxes[:, 4:5]
    all_h = boxes[:, 5:6]
    all_ry = boxes[:, 6:7]
    array0 = np.zeros([num_all, 1])
    all_cns[:, 0, :] = np.concatenate([-all_l/2.0, all_w/2.0, array0], axis=1)
    all_cns[:, 1, :] = np.concatenate([ all_l/2.0, all_w/2.0, array0], axis=1)
    all_cns[:, 2, :] = np.concatenate([ all_l/2.0,-all_w/2.0, array0], axis=1)
    all_cns[:, 3, :] = np.concatenate([-all_l/2.0,-all_w/2.0, array0], axis=1)
    all_cns[:, 4, :] = all_cns[:, 0, :] + np.concatenate([array0, array0, all_h], axis=1)
    all_cns[:, 5, :] = all_cns[:, 1, :] + np.concatenate([array0, array0, all_h], axis=1)
    all_cns[:, 6, :] = all_cns[:, 2, :] + np.concatenate([array0, array0, all_h], axis=1)
    all_cns[:, 7, :] = all_cns[:, 3, :] + np.concatenate([array0, array0, all_h], axis=1)
    for i in range(all_cns.shape[0]):
        all_cns[i, :, :] = apply_R(all_cns[i, :, :], rotz(all_ry[i]))
        all_cns[i, :, :] = apply_tr(all_cns[i, :, :], np.array([all_x[i], all_y[i], all_z[i]]).reshape(1,3))
    all_cns2d = all_cns[:, :4, :2]
    all_plg2d = []
    for cns in all_cns:
        all_plg2d.append(Polygon(cns.tolist()).buffer(0)) # buffer is to clean bowties
    return all_plg2d

def corner_to_standup_box2d(boxes_corner):
    '''
    convert box2d represented by four corners into standup_box2d [x1, y1, x2, y2]
    inputs:
        boxes_corner (np.array) [#boxes, 4, 2]
    returns:
        standup_boxes2d (np.array) [#boxes, 4] [x1, y1, x2, y2]
    reference:
        https://github.com/qianguih/voxelnet
    '''
    # (N, 4, 2) -> (N, 4) x1, y1, x2, y2
    N = boxes_corner.shape[0]
    standup_boxes2d = np.zeros((N, 4))
    standup_boxes2d[:, 0] = np.min(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 1] = np.min(boxes_corner[:, :, 1], axis=1)
    standup_boxes2d[:, 2] = np.max(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 3] = np.max(boxes_corner[:, :, 1], axis=1)

    return standup_boxes2d

if __name__ == "__main__":
    box =    np.array( [1, 2, 0, 2, 4, 1, 45 /180.0 * np.pi]).reshape(1, -1)
    others = np.array([[3, 0, 0, 4, 2, 1, 0],
                       [6, 6, 0, 2, 2, 1, 0],
                       [0, 3, 0, 3, 3, 1, 0]])
    # inter = compute_intersec(box, others, mode='2d-rot')
    boxes = np.vstack([box, others])
    scores = np.array([1, 0.6, 0.7, 0.8])
    idx = nms_general(boxes, scores, 0.05, mode='2d-rot')
    
    from matplotlib import pyplot as plt
    from descartes import PolygonPatch
    all_plg2d = boxes3d2polygon(boxes)
    # fig = plt.figure(1, figsize=(5,5), dpi=90)
    # ax = fig.add_subplot(111)
    # for plg in all_plg2d:
    #     ax.add_patch(PolygonPatch(plg))
    # plt.axis("auto")
    # plt.savefig("/usr/app/vis/all_box.png")
    fig = plt.figure(1, figsize=(5,5), dpi=90)
    ax = fig.add_subplot(111)
    for i in idx:
        ax.add_patch(PolygonPatch(all_plg2d[i]))
    plt.axis("auto")
    plt.savefig("/usr/app/vis/after_nms.png")    