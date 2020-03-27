'''
 File Created: Thu Mar 19 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
'''
from enum import Enum
import math
import numpy as np
from numpy.linalg import inv
from det3.dataloader.basedata import BaseFrame, BaseCalib, BaseObj
from det3.ops import read_txt, apply_T

class UdiFrame(BaseFrame):
    Frame = Enum('Frame', ('BASE', 'LIDARTOP', 'LIDARFRONT', 'LIDARLEFT', 'LIDARRIGHT'))

    def all_frames():
        return 'BASE, LIDARTOP, LIDARFRONT, LIDARLEFT, LIDARRIGHT'

class UdiCalib(BaseCalib):
    @staticmethod
    def _calib_file_key_to_frame(s):
        map_dict = {
            "Tr_base_to_lidar_top": (UdiFrame("LIDARTOP"), UdiFrame("BASE")),
            "Tr_base_to_lidar_front": (UdiFrame("LIDARFRONT"), UdiFrame("BASE")),
            "Tr_base_to_lidar_left": (UdiFrame("LIDARLEFT"), UdiFrame("BASE")),
            "Tr_base_to_lidar_right": (UdiFrame("LIDARRIGHT"), UdiFrame("BASE")),
            "Tr_lidar_top_to_base": (UdiFrame("BASE"), UdiFrame("LIDARTOP")),
            "Tr_lidar_front_to_base": (UdiFrame("BASE"), UdiFrame("LIDARFRONT")),
            "Tr_lidar_left_to_base": (UdiFrame("BASE"), UdiFrame("LIDARLEFT")),
            "Tr_lidar_right_to_base": (UdiFrame("BASE"), UdiFrame("LIDARRIGHT")),
            "Tr_base_to_base": (UdiFrame("BASE"), UdiFrame("BASE"))
        }
        return map_dict[s]

    def read_calib_file(self):
        '''
        '''
        calib = dict()
        str_list = read_txt(self._path)
        str_list = [itm.rstrip() for itm in str_list if itm != '\n']
        for itm in str_list:
            calib[UdiCalib._calib_file_key_to_frame(itm.split(':')[0])] = itm.split(':')[1]
        for k, v in calib.items():
            calib[k] = np.array([float(itm) for itm in v.split()]).astype(np.float32).reshape(4, 4)
        inv_calib = dict()
        for k, v in calib.items():
            inv_calib[(k[1], k[0])] = inv(v)
        calib[(UdiFrame("BASE"), UdiFrame("BASE"))] = np.eye(4).astype(np.float32)
        calib = {**calib, **inv_calib}
        self._data = calib
        return self
    
    def transform(self, pts, source_frame: UdiFrame, target_frame: UdiFrame):
        '''
        transform pts from source_frame to target_frame
        @pts: np.array [N, 3]
        @source_frame: UdiFrame
        @target_frame: UdiFrame
        -> pts_t: np.array [N, 3]
        '''
        # get T_ego_s
        T_ego_s = self._data[(source_frame, UdiFrame("BASE"))]
        # get T_t_ego
        T_t_ego = self._data[(UdiFrame("BASE"), target_frame)]
        # get T_t_s = T_t_ego X T_ego_s
        T = T_t_ego @ T_ego_s
        return apply_T(pts[:, :3], T)

class UdiObj(BaseObj):
    def __init__(self, arr=np.zeros(7), cls=None, score=None, frame="BASE"):
        super().__init__(arr, cls, score)
        self._current_frame = UdiFrame(frame)
    
    def __str__(self):
        score = "" if self.score is None else f" {self.score:.2f}"
        return f"{self.cls} {self.x:.2f} {self.y:.2f} {self.z:.2f} {self.l:.2f} {self.w:.2f} {self.h:.2f} {self.theta:.2f}{score}"

    def equal(self, other, acc_cls=None, atol=1e-2):
        acc_cls = [other.cls] if acc_cls is None else acc_cls
        return (self.current_frame == other.current_frame and
                self.cls in acc_cls and
                other.cls in acc_cls and
                np.isclose(self.h, other.h, atol) and
                np.isclose(self.l, other.l, atol) and
                np.isclose(self.w, other.w, atol) and
                np.isclose(self.x, other.x, atol) and
                np.isclose(self.y, other.y, atol) and
                np.isclose(self.z, other.z, atol) and
                np.isclose(math.cos(2 * (self.theta - other.theta)), 1, atol))
