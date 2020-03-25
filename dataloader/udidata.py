'''
 File Created: Thu Mar 19 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
'''
from enum import Enum
from det3.dataloader.basedata import BaseFrame, BaseCalib, BaseObj

class UdiFrame(BaseFrame):
    Frame = Enum('Frame', ('Ego'))

    def all_frames():
        return "Ego"

class UdiCalib(BaseCalib):
    def read_calib_file(self):
        '''
        TODO: Current point clouds and label are all in the Ego Frame.
        We need to split the merged LiDAR into several single LiDAR pc.
        '''
        raise NotImplementedError

class UdiObj(BaseObj):
    def __init__(self, arr, cls, score):
        super(self).__init__(arr, cls, score)
    
    def 
