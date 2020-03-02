'''
 File Created: Sat Feb 29 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
'''
import torch
import numpy as np
from numba import njit

@njit
def compute_intersect_2d_npy(box, others):
    '''
    compute the intersection between box and others under 2D aligned boxes.
    @box: np.ndarray (4,)
        [x, y, l, w] (x, y) is the center coordinate;
        l and w are the scales along x- and y- axes.
    @others: same to box (M, 4)
        [[x, y, l, w],...]
    -> its: intersection results with same type as box (M, )
    Note: under njit implementation: others(100boxes) -> 2ms for 1000 times
    '''
    M = others.shape[0]
    box_x, box_y, box_l, box_w = box.flatten()
    box_xmin, box_xmax = box_x - box_l/2.0, box_x + box_l/2.0
    box_ymin, box_ymax = box_y - box_w/2.0, box_y + box_w/2.0
    others_x, others_y = others[:, 0], others[:, 1]
    others_l, others_w = others[:, 2], others[:, 3]
    others_xmin, others_xmax = others_x - others_l/2.0, others_x + others_l/2.0
    others_ymin, others_ymax = others_y - others_w/2.0, others_y + others_w/2.0
    xx1 = np.maximum(box_xmin, others_xmin)
    yy1 = np.maximum(box_ymin, others_ymin)
    xx2 = np.minimum(box_xmax, others_xmax)
    yy2 = np.minimum(box_ymax, others_ymax)
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    return inter

def compute_intersect_2d_torch(box, others):
    '''
    compute the intersection between box and others under 2D aligned boxes.
    @box: torch.Tensor (4,)
        [x, y, l, w] (x, y) is the center coordinate;
        l and w are the scales along x- and y- axes.
    @others: same to box (M, 4)
        [[x, y, l, w],...]
    -> its: intersection results with same type as box (M, )
    Note: others(100boxes) -> 66ms for 1000 times
    '''
    import iou_cpp
    return iou_cpp.compute_intersect_2d(box, others)

def compute_intersect_2d_torchcuda(box, others):
    '''
    compute the intersection between box and others under 2D aligned boxes.
    @box: torch.Tensor.cuda (4,)
        [x, y, l, w] (x, y) is the center coordinate;
        l and w are the scales along x- and y- axes.
    @others: same to box (M, 4)
        [[x, y, l, w],...]
    -> its: intersection results with same type as box (M, )
    Note: others(100boxes) -> 200ms for 1000 times (w.o. counting cpu-gpu transfering time)
    '''
    return compute_intersect_2d_torch(box, others)

