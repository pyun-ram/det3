from easydict import EasyDict as edict
__C = edict()
cfg = __C

__C.Task = {
    "disp_itv":10,
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
        "name": "KittiDataset",
        "kitti_info_path": "/usr/app/data/KITTI/kitti_infos_train.pkl",
        "kitti_root_path": "/usr/app/data/KITTI/",
    },
    "DBSampler": {
        "name": "DataBaseSamplerV2",
        "db_info_path": "/usr/app/data/KITTI/kitti_dbinfos_train.pkl",
        "sample_groups": [
            {"Car": 15},
        ],
        "DBProcer": [
            {"name": "DBFilterByMinNumPoint",
             "min_gt_point_dict": {"Car": 5}},
            {"name": "DBFilterByDifficulty",
             "removed_difficulties": [-1]},
        ],
        "rate": 1.0,
        "global_random_rotation_range_per_object": [0, 0]
    },
    "PreProcess":{
        "max_number_of_voxels": 17000,
        "remove_unknown_examples": False,
        "shuffle_points": True,
        "groundtruth_rotation_uniform_noise": [-0.78539816, 0.78539816],
        "groundtruth_localization_noise_std": [1.0, 1.0, 0.5],
        "global_rotation_uniform_noise": [-0.78539816, 0.78539816],
        "global_scaling_uniform_noise": [0.95, 1.05],
        "global_random_rotation_range_per_object": [0, 0],
        "global_translate_noise_std": [0, 0, 0],
        "anchor_area_threshold": -1,
        "groundtruth_points_drop_percentage": 0.0,
        "groundtruth_drop_max_keep_points": 15,
        "remove_points_after_sample": True,
        "remove_environment": False,
        "use_group_id": False,
        "min_num_of_points_in_gt": -1, # deactivate
        "random_flip_x": False,
        "random_flip_y": True,
        "sample_importance": 1.0,
    }
}
__C.ValDataLoader = {
    "batch_size": 8,
    "num_workers": 8,
    "Dataset": {
        "name": "KittiDataset",
        "kitti_info_path": "/usr/app/data/KITTI/kitti_infos_val.pkl",
        "kitti_root_path": "/usr/app/data/KITTI/",
    },
    "PreProcess":{
        "max_number_of_voxels": 40000,
        "shuffle_points": False,
        "anchor_area_threshold": -1,
        "remove_environment": False,

        "remove_unknown_examples": None,
        "groundtruth_rotation_uniform_noise": [],
        "groundtruth_localization_noise_std": [],
        "global_rotation_uniform_noise": [],
        "global_scaling_uniform_noise":[],
        "global_random_rotation_range_per_object":[],
        "global_translate_noise_std":[],
        "groundtruth_points_drop_percentage":None,
        "groundtruth_drop_max_keep_points":None,
        "remove_points_after_sample":None,
        "use_group_id":None,
        "min_num_of_points_in_gt":None,
        "random_flip_x":None,
        "random_flip_y":None,
        "sample_importance":None,
    }
}
__C.Optimizer = {
    "name": "ADAMOptimizer",
    "amsgrad": False,
    "weight_decay": 0.01,
    "fixed_weight_decay": True,
    "steps": 23200 #464 * 50
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
    "restore": 'methods/second/saved_weights/Second-dev-000A/VoxelNet-22526.tckpt'
}