'''
 File Created: Sat Feb 29 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
 Note: This file includes the APIs of det3 operations.
    All the functions have to be well documented.
    The document of each API should include:
    - function
    - input (type) description
    - output (type) description
    - Special Note: If not specified, the function should be safe.
    i.e. not changing the inputs.
'''
# I/O
def read_txt(path:str):
    raise NotImplementedError

def write_txt(obj:list, path:str):
    raise NotImplementedError

def read_npy(path:str):
    raise NotImplementedError

def write_npy(obj, path:str):
    '''
    @obj: np.ndarray
    '''
    raise NotImplementedError

def read_pcd(path:str):
    raise NotImplementedError

def write_pcd(obj, path:str):
    '''
    @obj: np.ndarray
    '''
    raise NotImplementedError

def read_bin(path:str):
    raise NotImplementedError

def write_bin(obj, path:str):
    '''
    @obj: np.ndarray
    '''
    raise NotImplementedError

def read_img(path:str):
    raise NotImplementedError

def write_img(obj:np.ndarray, path:str):
    raise NotImplementedError

def read_pkl(path:str):
    raise NotImplementedError

def write_pkl(obj, path:str):
    '''
    @obj: any python object
    '''
    raise NotImplementedError

# IoU Computing
def compute_intersect_2d(box, others):
    '''
    compute the intersection between box and others under 2D aligned boxes.
    @box: np.ndarray/torch.Tensor/torch.Tensor.cuda (4,)
        [x, y, l, w] (x, y) is the center coordinate;
        l and w are the scales along x- and y- axes.
    @others: same to box (M, 4)
        [[x, y, l, w],...]
    -> its: intersection results with same type as box (M, )
    '''
    raise NotImplementedError

def compute_intersect_2drot(box, others):
    '''
    compute the intersection between box and others under 2D rotated boxes.
    @box: np.ndarray/torch.Tensor/torch.Tensor.cuda (5,)
        [x, y, l, w, theta] (x, y) is the center coordinate;
        l and w are the scales along x- and y- axes.
        theta is the rotation angle along the z-axis (counter-clockwise).
    @others: same to box (M, 5)
        [[x, y, l, w, theta],...]
    -> its: intersection results with same type as box (M, )
    '''
    raise NotImplementedError

def compute_intersect_3drot(box, others):
    '''
    compute the intersection between box and others under 3D rotated boxes.
    @box: np.ndarray/torch.Tensor/torch.Tensor.cuda (7,)
        [x, y, z, l, w, h, theta] (x, y, z) is the bottom center coordinate;
        l, w, and h are the scales along x-, y-, and z- axes.
        theta is the rotation angle along the z-axis (counter-clockwise).
    @others: same to box (M, 7)
        [[x, y, z, l, w, h, theta],...]
    -> its: intersection results with same type as box (M, )
    '''
    raise NotImplementedError

# NMS
def nms_2d(boxes, scores, thr_iou:float):
    '''
    Non-maximum surppression with 2D aligned boxes.
    @boxes: np.ndarray/torch.Tensor/torch.Tensor.cuda (M, 4)
        [[x, y, l, w]...] (x, y) is the center coordinate;
        l and w are the scales along x- and y- axes.
    @scores: same to boxes (M, )
        scores correspond to boxes
    @thr_iou: nms iou threshold within range (0, 1)
    -> keep: keeped index of boxes (M', ) M' <= M
    '''
    raise NotImplementedError

def nms_2drot(boxes, scores, thr_iou:float):
    '''
    Non-maximum surppression with 2D rotated boxes.
    @boxes: np.ndarray/torch.Tensor/torch.Tensor.cuda (M, 5)
        [[x, y, l, w, theta]...] (x, y) is the center coordinate;
        l and w are the scales along x- and y- axes.
        theta is the rotation angle along the z-axis (counter-clockwise).
    @scores: same to boxes (M, )
        scores correspond to boxes
    @thr_iou: nms iou threshold within range (0, 1)
    -> keep: keeped index of boxes (M', ) M' <= M
    '''
    raise NotImplementedError

def nms_3drot(boxes, scores, thr_iou:float):
    '''
    Non-maximum surppression with 2D rotated boxes.
    @boxes: np.ndarray/torch.Tensor/torch.Tensor.cuda (M, 7)
        [[x, y, z, l, w, h, theta]...] (x, y, z) is the bottom center coordinate;
        l, w, and h are the scales along x-, y-, and z- axes.
        theta is the rotation angle along the z-axis (counter-clockwise).
    @scores: same to boxes (M, )
        scores correspond to boxes
    @thr_iou: nms iou threshold within range (0, 1)
    -> keep: keeped index of boxes (M', ) M' <= M
    '''
    raise NotImplementedError

# 3D Point Transformation
def apply_tr(pts, tr_vec):
    '''
    apply translation on pts.
    @pts: np.ndarray/torch.Tensor/torch.Tensor.cuda (N, 3)
        [[x, y, z]...]
    @tr_vec: same to pts (3, )
        [tr_x, tr_y, tr_z]
    -> pts_tr same to pts (N, 3)
    Note: This function should be safe.
    '''
    raise NotImplementedError

def apply_R(pts, R_matrix):
    '''
    apply rotation on pts.
    @pts: np.ndarray/torch.Tensor/torch.Tensor.cuda (N, 3)
        [[x, y, z]...]
    @tr_vec: same to pts (3, 3)
        3x3 rotation matrix
    -> pts_R same to pts (N, 3)
    Note: This function should be safe.
    '''
    raise NotImplementedError

def hfill_pts(pts):
    '''
    convert pts to homogeneous coordinate.
    @pts: np.ndarray/torch.Tensor/torch.Tensor.cuda (N, 3)
    [[x, y, z]...]
    -> pts_h same to pts (N, 4)
    [[x, y, z, 1]...]
    '''
    raise NotImplementedError

# 3D Box Cropping
def crop_pts(boxes, pts):
    '''
    crop pts acoording to the boxes.
    @boxes: np.ndarray/torch.Tensor/torch.Tensor.cuda (M, 7)
            [[x, y, z, l, w, h, theta]...] (x, y, z) is the bottom center coordinate;
            l, w, and h are the scales along x-, y-, and z- axes.
            theta is the rotation angle along the z-axis (counter-clockwise).
    @pts: np.ndarray/torch.Tensor/torch.Tensor.cuda (N, 3)
    [[x, y, z]...]
    -> idxes: list [np.long/torch.long/torch.long.cuda, ...]
        assert len (idxes) == M
    '''
    raise NotImplementedError
