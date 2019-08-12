'''
File Created: Thursday, 8th August 2019 2:49:02 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
from easydict import EasyDict as edict
import math
import numpy as np
__C = edict()
cfg = __C

cfg.task_dict = {
    "TAG": "PointNet-Seg-000A",
    "GPU": 0,

}

cfg.data_dict = {
    "DATA_DIR": "/usr/app/data/ShapeNet/",
    "NUM_POINTS": 2500,
    "CLASS_CHOICE": "Chair",
}
cfg.model_dict = {
    "K": 16,
    "FEATURE_TRANSFORM": True,   
}

cfg.train_dict = {
    "SEED": None,
    "EPOCHS": 10,
    "START_EPOCH": 0,
    "BATCH_SIZE": 64,
    "NUM_DATALOAD_WKERS": 8,
    "LOG_GRAD": False,
    "LOG_ACTV": False,
    "LR_DICT": {
        "TYPE": "decay",
        "LR": 0.001,
        "STEP_SIZE": 20,
        "GAMMA": 0.5,
    },
    "RESUME": None,
    "VAL_FREQ": 1,

}