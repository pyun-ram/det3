from easydict import EasyDict as edict
import numpy as np
__C = edict()
cfg = __C

def deg2rad(deg):
    return deg / 180 * np.pi

__C.Task = {
    "disp_itv": 10,
    "save_itv": 901,
}
__C.Voxelizer = {
    "type": "VoxelizerV1",
    "voxel_size": [0.05, 0.05, 0.1],
    "point_cloud_range": [0, -40, -3, 70.4, 40, 1],
    "max_num_points": 5,
    "max_voxels": 20000
}
__C.BoxCoder = {
    "type": "BoxCoderV1",
    "custom_ndim": 0,
}
__C.AnchorGenerator = {
    "type": "AnchorGeneratorBEV",
    "class_name": "Car",
    "anchor_ranges": [0, -40.0, -1.00, 70.4, 40.0, -1.00],
    "sizes": [1.6, 3.9, 1.56], # wlh
    "rotations": [0, 1.57],
    "match_threshold": 0.6,
    "unmatch_threshold": 0.45,
}
__C.SimilarityCalculator = {
    "type": "NearestIoUSimilarity"
}
__C.TargetAssigner = {
    "type": "TaskAssignerV1",
    "classes": ["Car"],
    "feature_map_sizes": None,
    "region_similarity_calculators": ["nearest_iou_similarity"],
    "positive_fraction": None,
    "sample_size": 512,
    "assign_per_class": True,
}
__C.Net = {
    "name": "VoxelNet",
    "VoxelEncoder": {
        "name": "SimpleVoxel",
        "num_input_features": 4,
    },
    "MiddleLayer":{
        "name": "SpMiddleFHD",
        "use_norm": True,
        "num_input_features": 4,
        "downsample_factor": 8
    },
    "RPN":{
        "name": "RPNV2",
        "use_norm": True,
        "use_groupnorm": False,
        "num_groups": 0,
        "layer_nums": [5],
        "layer_strides": [1],
        "num_filters": [128],
        "upsample_strides": [1],
        "num_upsample_filters": [128],
        "num_input_features": 128,
    },
    "ClassificationLoss":{
        "name": "SigmoidFocalClassificationLoss",
        "alpha": 0.25,
        "gamma": 2.0,
    },
    "LocalizationLoss":{
        "name": "WeightedSmoothL1LocalizationLoss",
        "sigma": 3.0,
        "code_weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "codewise": True,
    },
    "num_class": 2,
    "use_sigmoid_score": True,
    "encode_background_as_zeros": True,
    "use_direction_classifier": True,
    "num_direction_bins": 2,
    "encode_rad_error_by_sin": True,
    "post_center_range": [0, -40, -2.2, 70.4, 40, 0.8],
    "nms_class_agnostic": False,
    "direction_limit_offset": 1,
    "sin_error_factor": 1.0,
    "use_rotate_nms": True,
    "multiclass_nms": False,
    "nms_pre_max_sizes": [1000],
    "nms_post_max_sizes": [100],
    "nms_score_thresholds": [0.3], # 0.4 in submit, but 0.3 can get better hard performance
    "nms_iou_thresholds": [0.01],
    "cls_loss_weight": 1.0,
    "loc_loss_weight": 2.0,
    "loss_norm_type": "NormByNumPositives",
    "direction_offset": 0.0,
    "direction_loss_weight": 0.2,
    "pos_cls_weight": 1.0,
    "neg_cls_weight": 1.0,
}

__C.TrainDataLoader = {
    "batch_size": 8,
    "num_workers": 8,
    "Dataset": {
        "name": "MyKittiDataset",
        "kitti_info_path": "/usr/app/data/MyKITTI/KITTI_infos_train.pkl",
        "kitti_root_path": "/usr/app/data/MyKITTI/",
    },
    "DBSampler": {
        "name": "DataBaseSamplerV3",
        "db_info_path": "/usr/app/data/MyKITTI/KITTI_dbinfos_train.pkl",
        "sample_dict": {"Car": 15},
        "sample_param":{
            "mode": "random", # "random"
            "x_range": ("abs", [10, 70.4]), # Flidar
            "y_range": ("abs", [-40, 40]), # Flidar
            "z_range": None, # Flidar
            "ry_range": None
        },
        "DBProcer": [
            {"name": "DBFilterByMinNumPoint",
             "min_gt_point_dict": {"Car": 5}},
        ],
    },
    "PreProcess":{
        "max_number_of_voxels": 17000,
        "augment_dict": {
            "p_rot": 0.2,
            "dry_range": [-45 / 180.0 * np.pi, 45 / 180.0 * np.pi],
            "p_tr": 0.3,
            "dx_range": [-1, 1],
            "dy_range": [-1, 1],
            "dz_range": [-0.1, 0.1],
            "p_flip": 0.1,
            "p_keep": 0.4
        },
    }
}
__C.ValDataLoader = {
    "batch_size": 8,
    "num_workers": 8,
    "Dataset": {
        "name": "MyKittiDataset",
        "kitti_info_path": "/usr/app/data/MyKITTI/KITTI_infos_val.pkl",
        "kitti_root_path": "/usr/app/data/MyKITTI/",
    },
    "PreProcess":{
        "max_number_of_voxels": 40000,
        "augment_dict": None
    }
}
__C.Optimizer = {
    "name": "ADAMOptimizer",
    "amsgrad": False,
    "weight_decay": 0.01,
    "fixed_weight_decay": True,
    "steps" : 464 * 50
}
__C.LRScheduler = {
    "name": "OneCycle",
    "lr_max": 2.25e-3,
    "moms": [0.95, 0.85],
    "div_factor": 10.0,
    "pct_start": 0.4,
}
__C.Evaluater = {}
__C.WeightManager = {
    "restore": None,
}