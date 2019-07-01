'''
File Created: Wednesday, 26th June 2019 10:25:13 am
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import sys
sys.path.append("../")
from det3.dataloarder.kittidata import KittiLabel, KittiCalib, KittiObj
from det3.dataloarder.carladata import CarlaLabel, CarlaCalib, CarlaObj
from det3.utils.utils import istype, apply_R, apply_tr, rotz, compute_intersec
import numpy as np
from typing import List


class KittiAugmentor:
    def __init__(self, *, p_rot: float = 0, p_tr: float = 0, p_flip: float = 0, p_keep: float = 0,
                 dx_range: List[float] = None, dy_range: List[float] = None, dz_range: List[float] = None,
                 dry_range: List[float] = None):
        '''
        Data augmentor for Kitti dataset.
        inputs:
            p_rot, p_tr, p_flip are the probabilities for the methods.
        '''
        self.dataset = 'Kitti'
        assert 0 <= p_rot <= 1
        assert 0 <= p_tr <= 1
        assert 0 <= p_flip <= 1
        assert 0 <= p_keep <= 1
        self.p_rot = np.random.rand() * p_rot
        self.p_tr = np.random.rand() * p_tr
        self.p_flip = np.random.rand() * p_flip
        self.p_keep = np.random.rand() * p_keep
        list_mode = ["rotate_obj", "tr_obj", "flip_pc", "keep"]
        list_pr = [self.p_rot, self.p_tr, self.p_flip, self.p_keep]
        self.mode = list_mode[np.argmax(list_pr)]
        self.dx_range = dx_range
        self.dy_range = dy_range
        self.dz_range = dz_range
        self.dry_range = dry_range
        self.dict_params = {
            "rotate_obj": [dry_range],
            "tr_obj": [dx_range, dy_range, dz_range],
            "flip_pc": [],
            "keep": []
        }
        if np.sum(list_pr) == 0:
            self.mode = None
        # print(self.mode)

    def check_overlap(self, label: KittiLabel) -> bool:
        '''
        check if there is overlap in label.
        inputs:
            label: the result after agmentation
        return:
            bool: True if no overlap exists.
        '''
        assert istype(label, "KittiLabel")
        boxes = []
        for obj in label.data:
            boxes.append([obj.x, obj.z, -obj.y, obj.l, obj.w, obj.h, -obj.ry]) # Fcam to Flidar
        boxes = np.vstack(boxes)
        while boxes.shape[0] > 0:
            box = boxes[0:1, :]
            others = boxes[1:, :]
            inter = compute_intersec(box, others, mode='2d-rot')
            if inter.sum() > 0:
                return False
            boxes = others
        return True

    def apply(self, label: KittiLabel, pc: np.array, calib: KittiCalib) -> (KittiLabel, np.array):
        assert self.mode is not None
        func = getattr(self, self.mode)
        params = [label, pc, calib]
        params += self.dict_params[self.mode]
        assert not any(itm is None for itm in params)
        return func(*params)

    def rotate_obj(self, label: KittiLabel, pc: np.array, calib: KittiCalib, dry_range: List[float]) -> (KittiLabel, np.array):
        '''
        rotate object along the z axis in the LiDAR frame
        inputs:
            label: gt
            pc: [#pts, >= 3]
            calib:
            dry_range: [dry_min, dry_max] in radius
        returns:
            label_rot
            pc_rot
        Note: If the attemp times is over max_attemp, it will return the original copy
        '''
        assert istype(label, "KittiLabel")
        dry_min, dry_max = dry_range
        max_attemp = 10
        num_attemp = 0
        while True:
            num_attemp += 1
            # copy pc & label
            pc_ = pc.copy()
            label_ = KittiLabel()
            label_.data = []
            for obj in label.data:
                label_.data.append(KittiObj(str(obj)))
            if num_attemp > max_attemp:
                print("KittiAugmentor.rotate_obj: Warning: the attemp times is over {} times,"
                      "will return the original copy".format(max_attemp))
                return label_, pc_
            for obj in label_.data:
                # generate random number
                dry = np.random.rand() * (dry_max - dry_min) + dry_min
                # modify pc_
                idx = obj.get_pts_idx(pc_[:, :3], calib)
                bottom_Fcam = np.array([obj.x, obj.y, obj.z]).reshape(1, -1)
                bottom_Flidar = calib.leftcam2lidar(bottom_Fcam)
                pc_[idx, :3] = apply_tr(pc_[idx, :3], -bottom_Flidar)
                # obj.ry += dry is correspond to rotz(-dry)
                # since obj is in cam frame
                # pc_ is in LiDAR frame
                pc_[idx, :3] = apply_R(pc_[idx, :3], rotz(-dry))
                pc_[idx, :3] = apply_tr(pc_[idx, :3], bottom_Flidar)
                # modify obj
                obj.ry += dry
            if self.check_overlap(label_):
                break
        return label_, pc_

    def tr_obj(self, label: KittiLabel, pc: np.array, calib: KittiCalib,
               dx_range: List[float], dy_range: List[float], dz_range: List[float]) -> (KittiLabel, np.array):
        '''
        translate object in the LiDAR frame
        inputs:
            label: gt
            pc: [#pts, >= 3]
            calib:
            dx_range: [dx_min, dx_max] in LiDAR frame
            dy_range: [dy_min, dy_max] in LiDAR frame
            dz_range: [dz_min, dz_max] in LiDAR frame
        returns:
            label_tr
            pc_tr
        Note: If the attemp times is over max_attemp, it will return the original copy
        '''
        assert istype(label, "KittiLabel")
        dx_min, dx_max = dx_range
        dy_min, dy_max = dy_range
        dz_min, dz_max = dz_range
        max_attemp = 10
        num_attemp = 0
        while True:
            num_attemp += 1
            # copy pc & label
            pc_ = pc.copy()
            label_ = KittiLabel()
            label_.data = []
            for obj in label.data:
                label_.data.append(KittiObj(str(obj)))
            if num_attemp > max_attemp:
                print("KittiAugmentor.tr_obj: Warning: the attemp times is over {} times,"
                      "will return the original copy".format(max_attemp))
                return label_, pc_
            for obj in label_.data:
                # gennerate ramdom number
                dx = np.random.rand() * (dx_max - dx_min) + dx_min
                dy = np.random.rand() * (dy_max - dy_min) + dy_min
                dz = np.random.rand() * (dz_max - dz_min) + dz_min
                # modify pc_
                idx = obj.get_pts_idx(pc_[:, :3], calib)
                dtr = np.array([dx, dy, dz]).reshape(1, -1)
                pc_[idx, :3] = apply_tr(pc_[idx, :3], dtr)
                # modify obj
                bottom_Fcam = np.array([obj.x, obj.y, obj.z]).reshape(1, -1)
                bottom_Flidar = calib.leftcam2lidar(bottom_Fcam)
                bottom_Flidar = apply_tr(bottom_Flidar, dtr)
                bottom_Fcam = calib.lidar2leftcam(bottom_Flidar)
                obj.x, obj.y, obj.z = bottom_Fcam.flatten()
            if self.check_overlap(label_):
                break
        return label_, pc_

    def flip_pc(self, label: KittiLabel, pc: np.array, calib: KittiCalib) -> (KittiLabel, np.array):
        '''
        flip point cloud along the y axis of the Kitti Lidar frame
        inputs:
            label: ground truth
            pc: point cloud
            calib:
        '''
        assert istype(label, "KittiLabel")
        # copy pc & label
        pc_ = pc.copy()
        label_ = KittiLabel()
        label_.data = []
        for obj in label.data:
            label_.data.append(KittiObj(str(obj)))
        # flip point cloud
        pc_[:, 1] *= -1
        # modify gt
        for obj in label_.data:
            bottom_Fcam = np.array([obj.x, obj.y, obj.z]).reshape(1, -1)
            bottom_Flidar = calib.leftcam2lidar(bottom_Fcam)
            bottom_Flidar[0, 1] *= -1
            bottom_Fcam = calib.lidar2leftcam(bottom_Flidar)
            obj.x, obj.y, obj.z = bottom_Fcam.flatten()
            obj.ry *= -1
        return label_, pc_

    def keep(self, label: KittiLabel, pc: np.array, calib: KittiCalib) -> (KittiLabel, np.array):
        # copy pc & label
        pc_ = pc.copy()
        label_ = KittiLabel()
        label_.data = []
        for obj in label.data:
            label_.data.append(KittiObj(str(obj)))
        return label_, pc_

class CarlaAugmentor:
    def __init__(self, *, p_rot: float = 0, p_tr: float = 0, p_flip: float = 0, p_keep: float = 0,
                 dx_range: List[float] = None, dy_range: List[float] = None, dz_range: List[float] = None,
                 dry_range: List[float] = None):
        '''
        Data augmentor for CARLA dataset.
        inputs:
            p_rot, p_tr, p_flip are the probabilities for the methods.
        '''
        self.dataset = 'Carla'
        assert 0 <= p_rot <= 1
        assert 0 <= p_tr <= 1
        assert 0 <= p_flip <= 1
        assert 0 <= p_keep <= 1
        self.p_rot = np.random.rand() * p_rot
        self.p_tr = np.random.rand() * p_tr
        self.p_flip = np.random.rand() * p_flip
        self.p_keep = np.random.rand() * p_keep
        list_mode = ["rotate_obj", "tr_obj", "flip_pc", "keep"]
        list_pr = [self.p_rot, self.p_tr, self.p_flip, self.p_keep]
        self.mode = list_mode[np.argmax(list_pr)]
        self.dx_range = dx_range
        self.dy_range = dy_range
        self.dz_range = dz_range
        self.dry_range = dry_range
        self.dict_params = {
            "rotate_obj": [dry_range],
            "tr_obj": [dx_range, dy_range, dz_range],
            "flip_pc": [],
            "keep": []
        }
        if np.sum(list_pr) == 0:
            self.mode = None
        # print(self.mode)

    def check_overlap(self, label: CarlaLabel) -> bool:
        '''
        check if there is overlap in label.
        inputs:
            label: the result after agmentation
        return:
            bool: True if no overlap exists.
        '''
        assert istype(label, "CarlaLabel")
        boxes = []
        for obj in label.data:
            boxes.append([-obj.y, obj.x, obj.z, obj.w, obj.l, obj.h, obj.ry]) # FIMU to Flidar
        boxes = np.vstack(boxes)
        while boxes.shape[0] > 0:
            box = boxes[0:1, :]
            others = boxes[1:, :]
            inter = compute_intersec(box, others, mode='2d-rot')
            if inter.sum() > 0:
                return False
            boxes = others
        return True

    def apply(self, label: CarlaLabel, pc: np.array, calib: CarlaCalib) -> (CarlaLabel, np.array):
        assert self.mode is not None
        func = getattr(self, self.mode)
        params = [label, pc, calib]
        params += self.dict_params[self.mode]
        assert not any(itm is None for itm in params)
        return func(*params)

    def rotate_obj(self, label: CarlaLabel, pc: np.array, calib: CarlaCalib, dry_range: List[float]) -> (CarlaLabel, np.array):
        '''
        rotate object along the z axis in the LiDAR frame
        inputs:
            label: gt
            pc: [#pts, >= 3] in IMU frame
            calib:
            dry_range: [dry_min, dry_max] in radius
        returns:
            label_rot
            pc_rot
        Note: If the attemp times is over max_attemp, it will return the original copy
        '''
        assert istype(label, "CarlaLabel")
        assert istype(calib, "CarlaCalib")
        dry_min, dry_max = dry_range
        max_attemp = 10
        num_attemp = 0
        while True:
            num_attemp += 1
            # copy pc & label
            pc_ = pc.copy()
            label_ = CarlaLabel()
            label_.data = []
            for obj in label.data:
                label_.data.append(CarlaObj(str(obj)))
            if num_attemp > max_attemp:
                print("CarlaAugmentor.rotate_obj: Warning: the attemp times is over {} times,"
                      "will return the original copy".format(max_attemp))
                return label_, pc_
            for obj in label_.data:
                dry = np.random.rand() * (dry_max - dry_min) + dry_min
                # modify pc
                idx = obj.get_pts_idx(pc_[:, :3], calib)
                bottom_Fimu = np.array([obj.x, obj.y, obj.z]).reshape(1, -1)
                pc_[idx, :3] = apply_tr(pc_[idx, :3], -bottom_Fimu)
                pc_[idx, :3] = apply_R(pc_[idx, :3], rotz(dry))
                pc_[idx, :3] = apply_tr(pc_[idx, :3], bottom_Fimu)
                # modify obj
                obj.ry += dry
            if self.check_overlap(label_):
                break
        return label_, pc_

    def tr_obj(self, label: CarlaLabel, pc: np.array, calib: CarlaCalib,
               dx_range: List[float], dy_range: List[float], dz_range: List[float]) -> (CarlaLabel, np.array):
        '''
        translate object in the IMU frame
        inputs:
            label: gt
            pc: [#pts, >= 3] in imu frame
            calib:
            dx_range: [dx_min, dx_max] in imu frame
            dy_range: [dy_min, dy_max] in imu frame
            dz_range: [dz_min, dz_max] in imu frame
        returns:
            label_tr
            pc_tr
        Note: If the attemp times is over max_attemp, it will return the original copy
        '''
        assert istype(label, "CarlaLabel")
        assert istype(calib, "CarlaCalib")
        dx_min, dx_max = dx_range
        dy_min, dy_max = dy_range
        dz_min, dz_max = dz_range
        max_attemp = 10
        num_attemp = 0
        while True:
            num_attemp += 1
            # copy pc & label
            pc_ = pc.copy()
            label_ = CarlaLabel()
            label_.data = []
            for obj in label.data:
                label_.data.append(CarlaObj(str(obj)))
            if num_attemp > max_attemp:
                print("CarlaAugmentor.tr_obj: Warning: the attemp times is over {} times,"
                      "will return the original copy".format(max_attemp))
                return label_, pc_
            for obj in label_.data:
                dx = np.random.rand() * (dx_max - dx_min) + dx_min
                dy = np.random.rand() * (dy_max - dy_min) + dy_min
                dz = np.random.rand() * (dz_max - dz_min) + dz_min
                # modify pc
                idx = obj.get_pts_idx(pc_[:, :3], calib)
                dtr = np.array([dx, dy, dz]).reshape(1, -1)
                pc_[idx, :3] = apply_tr(pc_[idx, :3], dtr)
                # modify obj
                obj.x += dx
                obj.y += dy
                obj.z += dz
            if self.check_overlap(label_):
                break
        return label_, pc_

    def flip_pc(self, label: CarlaLabel, pc: np.array, calib: CarlaCalib) -> (CarlaLabel, np.array):
        '''
        flip point cloud along the y axis of the Kitti Lidar frame
        inputs:
            label: ground truth
            pc: point cloud
            calib:
        '''
        assert istype(label, "CarlaLabel")
        assert istype(calib, "CarlaCalib")
        # copy pc & label
        pc_ = pc.copy()
        label_ = CarlaLabel()
        label_.data = []
        for obj in label.data:
            label_.data.append(CarlaObj(str(obj)))
        # flip point cloud
        pc_[:, 1] *= -1
        # modify gt
        for obj in label_.data:
            obj.y *= -1
            obj.ry *= -1
        return label_, pc_

    def keep(self, label: CarlaLabel, pc: np.array, calib: CarlaCalib) -> (CarlaLabel, np.array):
        assert istype(label, "CarlaLabel")
        assert istype(calib, "CarlaCalib")
        # copy pc & label
        pc_ = pc.copy()
        label_ = CarlaLabel()
        label_.data = []
        for obj in label.data:
            label_.data.append(CarlaObj(str(obj)))
        return label_, pc_

if __name__ == "__main__":
    from det3.dataloarder.kittidata import KittiData, KittiCalib
    from det3.utils.utils import read_pc_from_bin
    from det3.visualizer.vis import BEVImage
    from det3.methods.voxelnet.config import cfg
    from PIL import Image
    import os
    for i in range(2000):
        print(i)
        tag = "003377"
        calib, img, label, pc = KittiData("/usr/app/data/KITTI/train/", idx=tag).read_data()

        bevimg = BEVImage(x_range=(0, 70), y_range=(-40, 40), grid_size=(0.05, 0.05))
        bevimg.from_lidar(pc, scale=1)
        for obj in label.data:
            bevimg.draw_box(obj, calib, bool_gt=True)
        bevimg_img = Image.fromarray(bevimg.data)
        bevimg_img.save(os.path.join('/usr/app/vis/train/', 'test_KittiAugmentor_rotobj_origin.png'))

        agmtor = KittiAugmentor(p_rot=cfg.aug_dict["p_rot"], p_tr=cfg.aug_dict["p_tr"],
                                p_flip=cfg.aug_dict["p_flip"],p_keep=cfg.aug_dict["p_keep"],
                                dx_range=cfg.aug_param["dx_range"], dy_range=cfg.aug_param["dy_range"],
                                dz_range=cfg.aug_param["dz_range"], dry_range=cfg.aug_param["dry_range"])
        label, pc = agmtor.apply(label, pc, calib)
        bevimg = BEVImage(x_range=(0, 70), y_range=(-40, 40), grid_size=(0.05, 0.05))
        bevimg.from_lidar(pc, scale=1)
        for obj in label.data:
            bevimg.draw_box(obj, calib, bool_gt=True)
        bevimg_img = Image.fromarray(bevimg.data)
        bevimg_img.save(os.path.join('/usr/app/vis/train/', 'test_KittiAugmentor_rotobj_result.png'))
