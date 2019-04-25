'''
File Created: Tuesday, 23rd April 2019 11:23:50 am
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import numpy as np
import math
import os
from numpy.linalg import inv
try:
    from ..utils import utils
except:
    # Run script python3 dataloader/carladata.py
    import sys
    sys.path.append("../")
    import det3.utils.utils as utils

class CarlaCalib:
    '''
    class storing CARLA calib data

              ^x
              |
        y<--.(z)        LiDAR Frame and IMU Frame

              ^ z
              |
           y (x)---> x Camera Frame
        The original point of IMU Frame is the same to Camera Frame in world Frame.
    '''
    def __init__(self, calib_path):
        self.path = calib_path
        self.data = None
        self.num_of_lidar = None
        self.P0 = np.array([[50, 0., 600, 0.],
                            [0., 50, 180, 0.],
                            [0., 0., 1, 0.]])
    def read_calib_file(self):
        '''
        read CARLA calib file
        '''
        calib = dict()
        with open(self.path, 'r') as f:
            str_list = f.readlines()
        str_list = [itm.rstrip() for itm in str_list if itm != '\n']
        for itm in str_list:
            calib[itm.split(':')[0]] = itm.split(':')[1]
        for k, v in calib.items():
            calib[k] = [float(itm) for itm in v.split()]
        self.data = calib
        self.num_of_lidar = sum(['Tr_imu_to_velo' in itm for itm in self.data.keys()])
        return self

    def lidar2imu(self, pts, key):
        '''
        convert pts in lidar(<key> frame) to imu frame
        inputs:
            pts (np.array): [#pts, 3]
                point cloud in lidar<key> frame
            key (str):  'Tr_imu_to_velo_XX',
                        It should be corresopnd to self.data.keys()
        returns:
            pts_imu (np.array): [#pts, 3]
                point cloud in imu frame
        '''
        if self.data is None:
            print("read_calib_file should be read first")
            raise RuntimeError
        assert pts.shape[1] == 3
        hfiller = np.expand_dims(np.ones(pts.shape[0]), axis=1)
        pts_hT = np.hstack([pts, hfiller]).T #(4, #pts)
        Tr_imu_to_velo = np.array(self.data[key]).reshape(4, 4)
        Tr_velo_to_imu = inv(Tr_imu_to_velo)
        pts_imu_T = Tr_velo_to_imu @ pts_hT # (4, #pts)
        pts_imu = pts_imu_T.T
        return pts_imu[:, :3]

    def imu2cam(self, pts):
        '''
        convert pts in imu frame to cam frame
        inputs:
            pts (np.array): [#pts, 3]
                point clouds in imu frame
            key (str):  'Tr_imu_to_velo_XX',
                        It should be corresopnd to self.data.keys()
        returns:
            pts_cam (np.array): [#pts, 3]
                point cloud in camera frame
        '''
        assert pts.shape[1] == 3
        pts_imu = pts
        pts_imu_x = pts_imu[:, 0:1]
        pts_imu_y = pts_imu[:, 1:2]
        pts_imu_z = pts_imu[:, 2:3]
        return np.hstack([-pts_imu_y, -pts_imu_z, pts_imu_x])

    def cam2imgplane(self, pts):
        '''
        project the pts from the camera frame to camera plane
        pixels = P2 @ pts_cam
        inputs:
            pts(np.array): [#pts, 3]
                points in camera frame
        return:
            pixels: [#pts, 2]
                pixels on the image
        Note: the returned pixels are floats
        '''
        if self.data is None:
            print("read_calib_file should be read first")
            raise RuntimeError
        hfiller = np.expand_dims(np.ones(pts.shape[0]), axis=1)
        pts_hT = np.hstack([pts, hfiller]).T #(4, #pts)
        pixels_T = self.P0 @ pts_hT #(3, #pts)
        pixels = pixels_T.T
        pixels[:, 0] /= pixels[:, 2]
        pixels[:, 1] /= pixels[:, 2]
        return pixels[:, :2]

class CarlaObj():
    '''
    class storing a Carla 3d object
    Defined in IMU frame
    '''
    def __init__(self, s=None):
        self.type = None
        self.truncated = None
        self.occluded = None
        self.alpha = None
        self.bbox_l = None
        self.bbox_t = None
        self.bbox_r = None
        self.bbox_b = None
        self.h = None
        self.w = None
        self.l = None
        self.x = None
        self.y = None
        self.z = None
        self.ry = None
        self.score = None
        if s is None:
            return
        if len(s.split()) == 15: # data
            self.truncated, self.occluded, self.alpha,\
            self.bbox_l, self.bbox_t, self.bbox_r, self.bbox_b, \
            self.h, self.w, self.l, self.x, self.y, self.z, self.ry = \
            [float(itm) for itm in s.split()[1:]]
            self.type = s.split()[0]
        elif len(s.split()) == 16: # result
            self.truncated, self.occluded, self.alpha,\
            self.bbox_l, self.bbox_t, self.bbox_r, self.bbox_b, \
            self.h, self.w, self.l, self.x, self.y, self.z, self.ry, self.score = \
            [float(itm) for itm in s.split()[1:]]
            self.type = s.split()[0]
        else:
            raise NotImplementedError

    def __str__(self):
        if self.score is None:
            return "{} {} {} {} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                self.type, self.truncated, self.occluded, self.alpha,\
                self.bbox_l, self.bbox_t, self.bbox_r, self.bbox_b, \
                self.h, self.w, self.l, self.x, self.y, self.z, self.ry)
        else:
            return "{} {} {} {} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                self.type, self.truncated, self.occluded, self.alpha,\
                self.bbox_l, self.bbox_t, self.bbox_r, self.bbox_b, \
                self.h, self.w, self.l, self.x, self.y, self.z, self.ry, self.score)

    def get_bbox3dcorners(self):
        '''
        get the 8 corners of the bbox3d in imu frame.
        1.--.2
         |  |
         |  |
        4.--.3 (bottom)

        5.--.6
         |  |
         |  |
        8.--.7 (top)

        IMU Frame:
                      ^x
                      |
                y<----.z
        '''
        # lwh <-> yxz (imu)
        l, w, h = self.l, self.w, self.h
        x, z, y = self.x, self.z, self.y
        bottom = np.array([
            [-l/2,  w/2, 0],
            [ l/2,  w/2, 0],
            [ l/2, -w/2, 0],
            [-l/2, -w/2, 0],
        ])
        bottom = utils.apply_R(bottom, utils.rotz(self.ry))
        bottom = utils.apply_tr(bottom, np.array([x, y, z]).reshape(-1, 3))
        top = utils.apply_tr(bottom, np.array([0, 0, h]))
        return np.vstack([bottom, top])

    def equal(self, obj, acc_cls, rtol):
        '''
        equal oprator for CarlaObj
        inputs:
            obj: CarlaObj
            acc_cls: list [str]
                ['Car', 'Van']
            eot: float
        Note: For ry, return True if obj1.ry == obj2.ry + n * pi
        '''
        assert isinstance(obj, CarlaObj)
        return (self.type in acc_cls and
                obj.type in acc_cls and
                np.isclose(self.h, obj.h, rtol) and
                np.isclose(self.l, obj.l, rtol) and
                np.isclose(self.w, obj.w, rtol) and
                np.isclose(self.x, obj.x, rtol) and
                np.isclose(self.y, obj.y, rtol) and
                np.isclose(self.z, obj.z, rtol) and
                np.isclose(math.cos(2 * (self.ry - obj.ry)), 1, rtol))

class CarlaLabel:
    '''
    class storing Carla 3d object detection label
        self.data ([CarlaObj])
    '''
    def __init__(self, label_path=None):
        self.path = label_path
        self.data = None

    def read_label_file(self, no_dontcare=True):
        '''
        read CARLA label file
        '''
        self.data = []
        with open(self.path, 'r') as f:
            str_list = f.readlines()
        str_list = [itm.rstrip() for itm in str_list if itm != '\n']
        for s in str_list:
            self.data.append(CarlaObj(s))
        if no_dontcare:
            self.data = list(filter(lambda obj: obj.type != "DontCare", self.data))
        return self

    def __str__(self):
        '''
        TODO: Unit TEST
        '''
        s = ''
        for obj in self.data:
            s += obj.__str__() + '\n'
        return s

    def equal(self, label, acc_cls, rtol):
        '''
        equal oprator for CarlaLabel
        inputs:
            label: CarlaLabel
            acc_cls: list [str]
                ['Car', 'Van']
            eot: float
        Notes: O(N^2)
        '''
        assert isinstance(label, CarlaLabel)
        if len(self.data) != len(label.data):
            return False
        if len(self.data) == 0:
            return True
        bool_list = []
        for obj1 in self.data:
            bool_obj1 = False
            for obj2 in label.data:
                bool_obj1 = bool_obj1 or obj1.equal(obj2, acc_cls, rtol)
            bool_list.append(bool_obj1)
        return any(bool_list)

    def isempty(self):
        '''
        return True if self.data = None or self.data = []
        '''
        return self.data is None or len(self.data) == 0

class CarlaData:
    '''
    class storing a frame of Carla data
    Notes:
        The dir should be:
        calib/
            xx.txt
        label_imu/
            xx.txt
        velo_xx/
            xx.pcd
        velo_xx/
        velo_xx/
    '''
    def __init__(self, root_dir, idx):
        '''
        inputs:
            root_dir(str): carla dataset dir
            idx(str %6d): data index e.g. "000000"
        '''
        self.calib_path = os.path.join(root_dir, "calib", idx+'.txt')
        self.label_path = os.path.join(root_dir, "label_imu", idx+'.txt')

        velodyne_list = os.listdir(root_dir)
        velodyne_list = [itm if 'velo' in itm.split('_') else None for itm in velodyne_list]
        self.velodyne_list = list(filter(lambda itm: itm is not None, velodyne_list))
        self.velodyne_paths = [os.path.join(root_dir, itm, idx+'.pcd') for itm in self.velodyne_list]

    def read_data(self):
        '''
        read data
        returns:
            calib(CarlaCalib)
            label(CarlaLabel)
            pc(dict):
                point cloud in Lidar <tag> frame.
                pc[tag] = np.array [#pts, 3], tag is the name of the dir saving velodynes
        '''
        calib = CarlaCalib(self.calib_path).read_calib_file()
        label = CarlaLabel(self.label_path).read_label_file()
        pc = dict()
        
        for k, v in zip(self.velodyne_list, self.velodyne_paths):
            assert k == v.split('/')[-2]
            pc[k] = utils.read_pc_from_pcd(v)
        return pc, label, calib

if __name__ == "__main__":
    from det3.visualizer.vis import BEVImage, FVImage
    from PIL import Image
    import os
    os.makedirs('/usr/app/vis/dev/bev/', exist_ok=True)
    os.makedirs('/usr/app/vis/dev/fv/', exist_ok=True)
    for i in range(100, 300):
        tag = "{:06d}".format(i)
        pc, label, calib = CarlaData('/usr/app/data/CARLA/dev/', tag).read_data()
        bevimg =  BEVImage(x_range=(-50, 50), y_range=(-50, 50), grid_size=(0.05, 0.05))
        point_cloud = np.vstack([calib.lidar2imu(pc['velo_top'], key='Tr_imu_to_velo_top'),
                                 calib.lidar2imu(pc['velo_left'], key='Tr_imu_to_velo_left'),
                                 calib.lidar2imu(pc['velo_right'], key='Tr_imu_to_velo_right'),
                                ])
        bevimg.from_lidar(point_cloud)
        for obj in label.data:
            bevimg.draw_box(obj, calib)
            print(obj)
        bevimg_img = Image.fromarray(bevimg.data)
        bevimg_img.save("/usr/app/vis/dev/bev/{}.png".format(tag))
        fvimg = FVImage()
        fvimg.from_lidar(calib, calib.lidar2imu(pc['velo_top'], key='Tr_imu_to_velo_top'))
        for obj in label.data:
            fvimg.draw_box(obj, calib)
            print(obj)
        fvimg_img = Image.fromarray(fvimg.data)
        fvimg_img.save('/usr/app/vis/dev/fv/{}.png'.format(tag))


