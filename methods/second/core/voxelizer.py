import numpy as np
class BaseVoxelizer:
    def __init__(self):
        raise NotImplementedError
    
    def apply(self, pc):
        raise NotImplementedError

class VoxelizerV1(BaseVoxelizer):
    '''
    Source: https://github.com/traveller59/second.pytorch
    '''
    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=20000):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        voxel_size = np.array(voxel_size, np.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)
        voxelmap_shape = tuple(np.round(grid_size).astype(np.int32).tolist())
        voxelmap_shape = voxelmap_shape[::-1]

        self._coor_to_voxelidx = np.full(voxelmap_shape, -1, dtype=np.int32)
        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size

    def apply(self, pc):
        res = self.points_to_voxel(pc, self._voxel_size,
                                   self._point_cloud_range,
                                   self._coor_to_voxelidx,
                                   self._max_num_points,
                                   self._max_voxels)
        for k, v in res.items():
            if k != "voxel_num":
                res[k] = v[:res["voxel_num"]]
        return res

    def points_to_voxel(self, points,
                        voxel_size,
                        coors_range,
                        coor_to_voxelidx,
                        max_points=35,
                        max_voxels=20000):
        """convert 3d points(N, >=3) to voxels. This version calculate
        everything in one loop. now it takes only 0.8ms(~6k voxels) 
        with c++ and 3.2ghz cpu.

        Args:
            points: [N, ndim] float tensor. points[:, :3] contain xyz points and
                points[:, 3:] contain other information such as reflectivity.
            voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
            coors_range: [6] list/tuple or array, float. indicate voxel range.
                format: xyzxyz, minmax
            coor_to_voxelidx: int array. used as a dense map.
            max_points: int. indicate maximum points contained in a voxel.
            max_voxels: int. indicate maximum voxels this function create.
                for voxelnet, 20000 is a good choice. you should shuffle points
                before call this function because max_voxels may drop some points.
            full_mean: bool. if true, all empty points in voxel will be filled with mean
                of exist points.
            block_filtering: filter voxels by height. used for lidar point cloud.
                use some visualization tool to see filtered result.
        Returns:
            voxels: [M, max_points, ndim] float tensor. only contain points.
            coordinates: [M, 3] int32 tensor. zyx format.
            num_points_per_voxel: [M] int32 tensor.
        """
        from spconv.spconv_utils import points_to_voxel_3d_np
        if not isinstance(voxel_size, np.ndarray):
            voxel_size = np.array(voxel_size, dtype=points.dtype)
        if not isinstance(coors_range, np.ndarray):
            coors_range = np.array(coors_range, dtype=points.dtype)
        voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
        voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
        voxelmap_shape = voxelmap_shape[::-1]
        num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
        voxels = np.zeros(
            shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
        voxel_point_mask = np.zeros(
            shape=(max_voxels, max_points), dtype=points.dtype)
        coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
        res = {
            "voxels": voxels,
            "coordinates": coors,
            "num_points_per_voxel": num_points_per_voxel,
            "voxel_point_mask": voxel_point_mask,
        }
        voxel_num = points_to_voxel_3d_np(
            points, voxels, voxel_point_mask, coors,
            num_points_per_voxel, coor_to_voxelidx, voxel_size.tolist(),
            coors_range.tolist(), max_points, max_voxels)
        res["voxel_num"] = voxel_num
        res["voxel_point_mask"] = res["voxel_point_mask"].reshape(
            -1, max_points, 1)
        return res

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def grid_size(self):
        return self._grid_size

if __name__=="__main__":
    from det3.utils.utils import read_pc_from_bin
    from spconv.utils import VoxelGeneratorV2
    pc = read_pc_from_bin('/usr/app/data/KITTI/dev/velodyne/000123.bin')
    voxel_size = [0.05, 0.05, 0.1]
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    max_num_points = 5
    max_voxels = 20000
    voxelizer = VoxelizerV1(voxel_size=voxel_size, point_cloud_range=point_cloud_range, max_num_points=max_num_points, max_voxels=max_voxels)
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

