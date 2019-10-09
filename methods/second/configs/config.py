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
__C.BoxCoder = {}
__C.Net = {}
__C.DataLoader = {}
__C.Optimizer = {}
__C.Evaluater = {}
__C.WeightManager = {}