from det3.methods.second.data.dataset import Dataset, register_dataset
from det3.utils.utils import load_pickle, read_pc_from_bin, read_image
from pathlib import Path

@register_dataset
class MyKittiDataset(Dataset):
    def __init__(self,
                 root_path,
                 info_path,
                 class_names,
                 prep_func,
                 num_point_features):
        # load info_file
        infos = load_pickle(info_path)
        self._kitti_infos = infos
        self._root_path = Path(root_path)
        self._class_names = class_names
        self._prep_func = prep_func
        self._num_point_features = num_point_features

    def __len__(self):
        return len(self._kitti_infos)

    def convert_detection_to_kitti_label(self, detection):
        # inherit from kitti_dataset.py
        # Note: the [x,y,z] here I used is the bottom center instead of the center
        raise NotImplementedError

    def evaluation(self, detections, output_dir):
        # inherit from kitti_dataset.py
        raise NotImplementedError

    def __getitem__(self, idx):
        input_dict = self.get_sensor_data(idx)
        example = self._prep_func(input_dict=input_dict)
        return example

    def get_sensor_data(self, query):
        idx = query
        info = self._kitti_infos[idx]
        calib = info["calib"]
        label = info["label"]
        tag = info["tag"]
        pc_reduced = read_pc_from_bin(info["reduced_pc_path"])
        res = {
            "lidar": {
                "points": pc_reduced,
            },
            "metadata": {
                "tag": tag
            },
            "calib": calib,
            "cam": {
                "label": label
            }
        }
        return res

if __name__ == "__main__":
    from det3.methods.second.data.mypreprocess import prep_pointcloud
    from functools import partial
    from det3.methods.second.configs.config import cfg
    from det3.methods.second.utils.import_tool import load_module
    from det3.methods.second.data.mypreprocess import DataBasePreprocessor
    from det3.methods.second.core.db_sampler import DataBaseSamplerV3
    from det3.methods.second.builder import (voxelizer_builder, box_coder_builder,
                                             similarity_calculator_builder, 
                                             anchor_generator_builder, target_assigner_builder)
    from det3.dataloader.augmentor import KittiAugmentor
    import numpy as np
    voxelizer = voxelizer_builder.build(voxelizer_cfg=cfg.Voxelizer)
    anchor_generator = anchor_generator_builder.build(anchor_generator_cfg=cfg.AnchorGenerator)
    box_coder = box_coder_builder.build(box_coder_cfg=cfg.BoxCoder)
    similarity_calculator = similarity_calculator_builder.build(similarity_calculator_cfg=cfg.SimilarityCalculator)
    target_assigner = target_assigner_builder.build(target_assigner_cfg=cfg.TargetAssigner,
                                                    box_coder=box_coder,
                                                    anchor_generators=[anchor_generator],
                                                    region_similarity_calculators=[similarity_calculator])
    dbprocers_cfg = cfg.TrainDataLoader["DBSampler"]["DBProcer"]
    tmps = [(load_module("methods/second/data/mypreprocess.py", cfg["name"]), cfg) for cfg in dbprocers_cfg]
    dbproces = []
    for builder, params in tmps:
        del params["name"]
        dbproces.append(builder(**params))
    db_prepor = DataBasePreprocessor(dbproces)
    db_infos = load_pickle("/usr/app/data/MyKITTI/KITTI_dbinfos_train.pkl")
    train_infos = load_pickle("/usr/app/data/MyKITTI/KITTI_infos_train.pkl")
    sample_dict = cfg.TrainDataLoader["DBSampler"]["sample_dict"]
    dbsampler = DataBaseSamplerV3(db_infos, db_prepor=db_prepor, sample_dict=sample_dict)
    # augmentor = KittiAugmentor(p_rot=0.2, p_flip=0.2, p_keep=0.4, p_tr=0.2,
    #                            dx_range=[-0.5, 0.5], dy_range=[-0.5, 0.5], dz_range=[-0.1, 0.1],
    #                            dry_range=[-5 * 180 / np.pi, 5 * 180 / np.pi])
    augment_dict={
        "p_rot": 0.2,
        "dry_range": [-5 * 180 / np.pi, 5 * 180 / np.pi],
        "p_tr": 0.3,
        "dx_range": [-0.5, 0.5],
        "dy_range": [-0.5, 0.5],
        "dz_range": [-0.1, 0.1],
        "p_flip": 0.2,
        "p_keep": 0.3
    }
    prep_cfg = cfg.TrainDataLoader["PreProcess"]
    prep_func = partial(prep_pointcloud,
                        training=True,
                        db_sampler=dbsampler,
                        augment_dict=augment_dict,
                        voxelizer=voxelizer,
                        max_voxels=prep_cfg["max_number_of_voxels"],
                        target_assigner=target_assigner)
    from det3.methods.second.ops.ops import rbbox2d_to_near_bbox
    grid_size = voxelizer.grid_size
    out_size_factor=2
    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]
    ret = target_assigner.generate_anchors(feature_map_size)
    class_names = target_assigner.classes
    anchors_dict = target_assigner.generate_anchors_dict(feature_map_size)
    anchors_list = []
    for k, v in anchors_dict.items():
        anchors_list.append(v["anchors"])
    
    # anchors = ret["anchors"]
    anchors = np.concatenate(anchors_list, axis=0)
    anchors = anchors.reshape([-1, target_assigner.box_ndim])
    assert np.allclose(anchors, ret["anchors"].reshape(-1, target_assigner.box_ndim))
    matched_thresholds = ret["matched_thresholds"]
    unmatched_thresholds = ret["unmatched_thresholds"]
    anchors_bv = rbbox2d_to_near_bbox(
        anchors[:, [0, 1, 3, 4, 6]])
    anchor_cache = {
        "anchors": anchors,
        "anchors_bv": anchors_bv,
        "matched_thresholds": matched_thresholds,
        "unmatched_thresholds": unmatched_thresholds,
        "anchors_dict": anchors_dict,
    }
    prep_func = partial(prep_func, anchor_cache=anchor_cache)
    kitti_data = MyKittiDataset(root_path="/usr/app/data/MyKITTI/training/",
                                info_path="/usr/app/data/MyKITTI/KITTI_infos_train.pkl",
                                class_names=["Car"],
                                prep_func=prep_func,
                                num_point_features=4)
    print(kitti_data[0])