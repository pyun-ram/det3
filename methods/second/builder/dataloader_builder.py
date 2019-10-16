import pickle
import numpy as np
from det3.methods.second.utils.import_tool import load_module
from det3.methods.second.utils.log_tool import Logger
from det3.methods.second.core.second import get_downsample_factor
from det3.methods.second.data.dataset import get_dataset_class
from det3.methods.second.data.preprocess import prep_pointcloud
from det3.methods.second.ops.ops import rbbox2d_to_near_bbox
from det3.methods.second.data.dataset import DatasetWrapper
from functools import partial

def build(model_cfg, dataloader_cfg, voxelizer, target_assigner, training):
    from det3.methods.second.data.preprocess import DataBasePreprocessor
    if "DBSampler" in dataloader_cfg.keys():
        # build DBProcer
        dbprocers_cfg = dataloader_cfg["DBSampler"]["DBProcer"]
        tmps = [(load_module("methods/second/data/preprocess.py", cfg["name"]), cfg) for cfg in dbprocers_cfg]
        dbproces = []
        for builder, params in tmps:
            del params["name"]
            dbproces.append(builder(**params))
        db_prepor = DataBasePreprocessor(dbproces)
        # build DBSampler
        info_path = dataloader_cfg["DBSampler"]["db_info_path"]
        with open(info_path, "rb") as f:
            db_infos = pickle.load(f)
        groups = dataloader_cfg["DBSampler"]["sample_groups"]
        rate = dataloader_cfg["DBSampler"]["rate"]
        global_rot_range = dataloader_cfg["DBSampler"]["global_random_rotation_range_per_object"]
        dbsampler_builder = load_module("methods/second/core/db_sampler.py", dataloader_cfg["DBSampler"]["name"])
        dbsampler = dbsampler_builder(db_infos=db_infos,
                                    groups=groups,
                                    db_prepor=db_prepor,
                                    rate=rate,
                                    global_rot_range=global_rot_range)
        Logger.log_txt("Warning: dataloader_builder.py: DBSampler build function should be changed to configurable.")
    else:
        dbsampler = None
        Logger.log_txt("Warning: dataloader_builder.py: DBSampler build function should be changed to configurable.")
    grid_size = voxelizer.grid_size
    out_size_factor = get_downsample_factor(model_cfg)
    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]
    assert all([n != '' for n in target_assigner.classes]), "you must specify class_name in anchor_generators."
    dataset_cls = get_dataset_class(dataloader_cfg["Dataset"]["name"])
    num_point_features = model_cfg["VoxelEncoder"]["num_input_features"]
    assert dataset_cls.NumPointFeatures >= 3, "you must set this to correct value"
    assert dataset_cls.NumPointFeatures == num_point_features, "currently you need keep them same"
    prep_cfg = dataloader_cfg["PreProcess"]
    prep_func = partial(
        prep_pointcloud,
        root_path=dataloader_cfg["Dataset"]["kitti_root_path"],
        voxel_generator=voxelizer,
        target_assigner=target_assigner,
        training=training,
        max_voxels=prep_cfg["max_number_of_voxels"],
        remove_outside_points=False,
        remove_unknown=prep_cfg["remove_unknown_examples"],
        create_targets=training,
        shuffle_points=prep_cfg["shuffle_points"],
        gt_rotation_noise=list(prep_cfg["groundtruth_rotation_uniform_noise"]),
        gt_loc_noise_std=list(prep_cfg["groundtruth_localization_noise_std"]),
        global_rotation_noise=list(prep_cfg["global_rotation_uniform_noise"]),
        global_scaling_noise=list(prep_cfg["global_scaling_uniform_noise"]),
        global_random_rot_range=list(
            prep_cfg["global_random_rotation_range_per_object"]),
        global_translate_noise_std=list(prep_cfg["global_translate_noise_std"]),
        db_sampler=dbsampler,
        num_point_features=dataset_cls.NumPointFeatures,
        anchor_area_threshold=prep_cfg["anchor_area_threshold"],
        gt_points_drop=prep_cfg["groundtruth_points_drop_percentage"],
        gt_drop_max_keep=prep_cfg["groundtruth_drop_max_keep_points"],
        remove_points_after_sample=prep_cfg["remove_points_after_sample"],
        remove_environment=prep_cfg["remove_environment"],
        use_group_id=prep_cfg["use_group_id"],
        out_size_factor=out_size_factor,
        multi_gpu=False,
        min_points_in_gt=prep_cfg["min_num_of_points_in_gt"],
        random_flip_x=prep_cfg["random_flip_x"],
        random_flip_y=prep_cfg["random_flip_y"],
        sample_importance=prep_cfg["sample_importance"])


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
    dataset = dataset_cls(
        info_path=dataloader_cfg["Dataset"]["kitti_info_path"],
        root_path=dataloader_cfg["Dataset"]["kitti_root_path"],
        class_names=class_names,
        prep_func=prep_func)
    dataset = DatasetWrapper(dataset)
    return dataset

if __name__ == "__main__":
    from det3.methods.second.builder import (voxelizer_builder, box_coder_builder,
                                         similarity_calculator_builder, 
                                         anchor_generator_builder, target_assigner_builder)
    cfg = load_module("methods/second/configs/config.py", name="cfg")
    voxelizer = voxelizer_builder.build(voxelizer_cfg=cfg.Voxelizer)
    anchor_generator = anchor_generator_builder.build(anchor_generator_cfg=cfg.AnchorGenerator)
    box_coder = box_coder_builder.build(box_coder_cfg=cfg.BoxCoder)
    similarity_calculator = similarity_calculator_builder.build(similarity_calculator_cfg=cfg.SimilarityCalculator)
    target_assigner = target_assigner_builder.build(target_assigner_cfg=cfg.TargetAssigner,
                                                    box_coder=box_coder,
                                                    anchor_generators=[anchor_generator],
                                                    region_similarity_calculators=[similarity_calculator])
    build(cfg.Net, cfg.TrainDataLoader, voxelizer, target_assigner, training=True)