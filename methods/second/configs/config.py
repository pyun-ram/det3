from easydict import EasyDict as edict
__C = edict()
cfg = __C

__C.Task = {
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
    "anchor_ranges": [0, -40.0, -1.00, 70.4, 40.0, -1.00],
    "sizes": [1.6, 3.9, 1.56], # wlh
    "rotations": [0, 1.57],
}
__C.SimilarityCalculator = {
    "type": "NearestIoUSimilarity"
}
__C.TargetAssigner = {
    "type": "TaskAssignerV1",
    "classes": ["Car"],
    "feature_map_sizes": [None],
    "region_similarity_calculators": ["nearest_iou_similarity"],
    "positive_fraction": -1,
    "sample_size": 512,
    "assign_per_class": True,
}

__C.Net = {}
__C.DataLoader = {}
__C.Optimizer = {}
__C.Evaluater = {}
__C.WeightManager = {}