from det3.methods.second.utils.import_tool import load_module

def build(voxelizer_cfg):
    class_name = voxelizer_cfg["type"]
    params = {k:v for k, v in voxelizer_cfg.items() if k != "type"}
    builder = load_module("methods/second/core/voxelizer.py", name=class_name)
    voxelizer = builder(**params)
    return voxelizer

if __name__ == "__main__":
    from det3.utils.utils import read_pc_from_bin
    from spconv.utils import VoxelGeneratorV2
    cfg = load_module("methods/second/configs/config.py", name="cfg")
    pc = read_pc_from_bin('/usr/app/data/KITTI/dev/velodyne/000123.bin')
    voxel_size = [0.05, 0.05, 0.1]
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    max_num_points = 5
    max_voxels = 20000
    voxelizer = build(cfg.Voxelizer)
    voxelizer_gt = VoxelGeneratorV2(voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels,
                 full_mean=False,
                 block_filtering=False,
                 block_factor=8,
                 block_size=3,
                 height_threshold=0.1,
                 height_high_threshold=2.0)
    res_est = voxelizer.apply(pc)
    res_gt = voxelizer_gt.generate(pc)
    for k in res_gt.keys():
        try:
            print(f"{k}: {(res_est[k] == res_gt[k]).all()}")
        except:
            print(f"{k}: {(res_est[k] == res_gt[k])}")

