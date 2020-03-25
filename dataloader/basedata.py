'''
 File Created: Thu Mar 19 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
 Note: This file contains the ABC for implementing Dataloader of 3D object detection datasets.
'''
from abc import ABC, abstractmethod
from enum import Enum

class BaseFrame(ABC):
    """
    This ABC is to define the frame.
    """
    Frame = Enum('Frame', ('Ego'))
    def __init__(self):
        self._frame = None

    @property
    def frame(self):
        assert self._frame is not None
        return self._frame
    
    @frame.setter
    def frame(self, value):
        '''
        @value: str/Enum
        '''
        if isinstance(value, str):
            self._current_frame = self.Frame[frame]
        elif isinstance(value, Enum):
            self._current_frame = frame
        else:
            raise TypeError("the type of value should be string or enum.")

    @staticmethod
    @abstractmethod
    def all_frames():
        '''
        ->return a string list all available frames
        '''
        raise NotImplementedError

    def __eq__(self, other):
        if isinstance(value, str):
            return self.frame == self.Frame[other]
        elif isinstance(value, Enum):
            return self.frame == other
        else:
            raise TypeError("the type of other should be string or enum.")

class BaseCalib(ABC):
    """
    This ABC is to define the calib class helping
    coordinate transformation between frames.
    """
    def __init__(self, path):
        self._path = path
        self._data = None

    @classmethod
    @abstractmethod
    def read_calib_file(self):
        '''
        Read the self._path to get calib information and write into self._data
        '''
        raise NotImplementedError

class BaseLabel(ABC):
    """
    This ABC is to define the label class helping manage the label.
    """
    def __init__(self, path):
        self._path = path
        self._objs = []
        self._objs_boxes = None
        self._objs_classes = []
        self._objs_scores = None
        self._current_frame = None
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def read_label_file(self):
        '''
        Read the label file and write into self._objs*.
        You have to set the self._current_frame.
        '''
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def boxes_order(self):
        '''
        ->return a string specifing the order of self._objs_boxes.
        e.g. "x, y, z, h, w, l, ry"
        '''
        raise NotImplementedError

    @property
    def boxes(self):
        return self._objs_boxes
    
    @property
    def scores(self):
        return self._objs_scores
    
    @property
    def classes(self):
        return self._objs_classes
    
    @property
    def current_frame(self):
        return self._current_frame
    
    @current_frame.setter
    def current_frame(self, value):
        self._current_frame = value
    
    def add_obj(self, obj):
        '''
        @obj: a derived class of BaseObj
        '''
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        '''
        ->return a derived class of BaseObj
        '''
        return self._objs[idx]

    def __eq__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

class BaseObj(ABC):
    """
    This ABC is to define the object helping manage one single object.
    """
    @classmethod
    @abstractmethod
    def __init__(self, arr, cls, score):
        '''
        @arr: np.array [7]
        [x, y, z] - bottom center
        [l, w, h] - scale: x- y- z- axis
        ry- rotation along y axis
        '''
        self.x = None
        self.y = None
        self.z = None
        self.l = None
        self.w = None
        self.h = None
        self.ry = None
        self.cls = cls
        self.score = score
        self._current_frame = None
        (self.x, self.y, self.z, 
            self.l, self.w, self.h, self.ry) = arr.flatten()

    @classmethod
    @abstractmethod
    def __str__(self):
        '''
        -> return a string representing the object
        '''
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    @property
    def current_frame(self):
        return self._current_frame
    
    @current_frame.setter
    def current_frame(self, value):
        self._current_frame = value

    def get_pts_idx(self, pc):
        '''
        Get the index of pts in the bounding box.
        @ pc: np.array, torch.Tensor, torch.Tensor.cuda
            The object has to be in a same frame of pc.
        -> return a list of idxes.
        '''
        raise NotImplementedError

    def get_bbox3d_corners(self):
        '''
        Get the corners defined by the bounding box
        1.--.2
         |  |                  ^x
         |  |                  |
        4.--.3 (bottom) y<----.z
        5.--.6
         |  |
         |  |
        8.--.7 (top)
        -> return a np.array [8, 3]
        '''
        raise NotImplementedError

    def from_corners(self, corners, cls, score, frame):
        '''
        Define the bounding box from the corners
        @ corners: np.ndarray [8, 3] with a same definition of BaseObj.get_bbox3d_corners()
        @ cls: str
        @ score: float [0, 1] / None
        @ frame: a class derived from BaseFrame
        '''
        raise NotImplementedError


class BaseData(ABC):
    """
    This ABC is to define the class helping read a package of a dataset.
    """
    @classmethod
    @abstractmethod
    def __init__(self, root_dir, tag):
        '''
        To init the root_dir and idx and other dirs, like LiDAR dirs...
        @ root_dir: str
        @ tag: str
        '''
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def read_data(self):
        '''
        To read data and return a dict
        -> return dict
        '''
        raise NotImplementedError

