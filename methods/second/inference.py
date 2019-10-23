'''
File Created: Monday, 7th October 2019 8:35:49 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
python3 methods/second/inference.py \
    --tag Secon-dev-000A \
    --cfg methods/second/configs/config.py \
    --rawdata-dir None \
    --vis-dir /usr/app/vis/demo \
    --is-val

python3 methods/second/inference.py \
    --tag Secon-dev-000A \
    --cfg methods/second/configs/config.py \
    --rawdata-dir /usr/app/data/KITTI-RAW/2011_09_26/2011_09_26_drive_0015_sync \
    --vis-dir /usr/app/vis/2011_09_26_drive_0015_sync
'''
import os
import sys
import time
import argparse
import torch
import torch.utils.data
import numpy as np
from pathlib import Path
import pickle
from shutil import copy
from tqdm import tqdm
from det3.methods.second.utils import Logger, load_module
from det3.methods.second.builder import (voxelizer_builder, box_coder_builder,
                                         similarity_calculator_builder, 
                                         anchor_generator_builder, target_assigner_builder,
                                         second_builder, dataloader_builder)
from det3.methods.second.core.model_manager import save_models, restore
from det3.methods.second.train import load_config_file
from det3.utils.utils import read_image, read_pc_from_bin
from det3.dataloader.kittidata import KittiCalib

def merge_second_batch(batch_list):
    from collections import defaultdict
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}
    for key, elems in example_merged.items():
        if key in [
                'voxels', 'num_points', 'num_gt', 'voxel_labels', 'gt_names', 'gt_classes', 'gt_boxes'
        ]:
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'metadata':
            ret[key] = elems
        elif key == "calib":
            ret[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            for k1, v1 in ret[key].items():
                ret[key][k1] = np.stack(v1, axis=0)
        elif key == 'coordinates':
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        elif key == 'metrics':
            ret[key] = elems
        else:
            ret[key] = np.stack(elems, axis=0)
    return ret


def _worker_init_fn(worker_id):
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)
    print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])

def read_calib(calib_dir:str):
    def load_data(path):
        data = dict()
        with open(path, 'r') as f:
            str_list = f.readlines()
        str_list = [itm.rstrip() for itm in str_list if itm != '\n']
        for itm in str_list:
            data[itm.split(':')[0]] = itm.split(':')[1]
        for k, v in data.items():
            if k == "calib_time":
                continue
            data[k] = [float(itm) for itm in v.split()]
        return data
    cam2cam_data = load_data(os.path.join(calib_dir, "calib_cam_to_cam.txt"))
    velo2cam_data = load_data(os.path.join(calib_dir, "calib_velo_to_cam.txt"))
    calib = KittiCalib(calib_path=None)
    R0_rect = np.zeros([4, 4])
    R0_rect[0:3, 0:3] = np.array(cam2cam_data['R_rect_00']).reshape(3, 3)
    R0_rect[3, 3] = 1
    
    Tr_velo_to_cam = np.zeros([4, 4])
    Tr_velo_to_cam[0:3, 0:3] = np.array(velo2cam_data['R']).reshape(3, 3)
    Tr_velo_to_cam[0:3, 3] = np.array(velo2cam_data['T']).reshape(3,)
    Tr_velo_to_cam[3, 3] = 1

    P2 = np.array(cam2cam_data['P_rect_02']).reshape(3,4)
    calib.data = {**cam2cam_data, **velo2cam_data}
    calib.R0_rect = R0_rect
    calib.Tr_velo_to_cam = Tr_velo_to_cam
    calib.P2 = P2
    return calib

def main(tag, cfg_path, is_val: bool, rawdata_dir, vis_dir):
    root_dir = Path(__file__).parent
    log_dir = Path(vis_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir)
    cfg = load_config_file(cfg_path=cfg_path, log_dir=log_dir, backup=False)

    # build net
    voxelizer = voxelizer_builder.build(voxelizer_cfg=cfg.Voxelizer)
    anchor_generator = anchor_generator_builder.build(anchor_generator_cfg=cfg.AnchorGenerator)
    box_coder = box_coder_builder.build(box_coder_cfg=cfg.BoxCoder)
    similarity_calculator = similarity_calculator_builder.build(similarity_calculator_cfg=cfg.SimilarityCalculator)
    target_assigner = target_assigner_builder.build(target_assigner_cfg=cfg.TargetAssigner,
                                                    box_coder=box_coder,
                                                    anchor_generators=[anchor_generator],
                                                    region_similarity_calculators=[similarity_calculator])
    net = second_builder.build(cfg=cfg.Net, voxelizer=voxelizer, target_assigner=target_assigner).cuda()
    # build dataloader
    val_data = dataloader_builder.build(cfg.Net, cfg.ValDataLoader,
                                          voxelizer, target_assigner, training=False)
    val_dataloader = torch.utils.data.DataLoader(
        val_data,
        batch_size=cfg.TrainDataLoader["batch_size"],
        shuffle=False,
        num_workers=cfg.TrainDataLoader["num_workers"],
        pin_memory=False,
        collate_fn=merge_second_batch,
        worker_init_fn=_worker_init_fn,
        drop_last=False)
    if cfg.WeightManager["restore"] is not None:
        restore(cfg.WeightManager["restore"], net)
    logger = Logger()
    t = time.time()
    net.eval()
    if is_val:
        detections = []
        result_path_step = Path(vis_dir)
        result_path_step.mkdir(parents=True, exist_ok=True)
        logger.log_txt("#################################")
        logger.log_txt("# EVAL")
        logger.log_txt("#################################")
        with torch.no_grad():
            for val_example in tqdm(val_dataloader):
                val_example = example_convert_to_torch(val_example, torch.float32)
                detection = net(val_example)
                detections += detection
        result_dict = val_data.dataset.evaluation(detections, str(result_path_step))
        for k, v in result_dict["results"].items():
            logger.log_txt("Evaluation {}".format(k))
            logger.log_txt(v)
        logger.log_metrics(result_dict["detail"], -1)
        detections = val_data.dataset.convert_detection_to_kitti_annos(detections)
        with open(result_path_step / "result.pkl", 'wb') as f:
            pickle.dump(detections, f)
    else:
        detections = []
        # load raw data
        data_dir = rawdata_dir
        vis_dir = vis_dir
        os.makedirs(vis_dir, exist_ok=True)
        pc_dir = os.path.join(data_dir, "velodyne_points", "data")
        img2_dir = os.path.join(data_dir, "image_02", "data")
        calib_dir = os.path.join(data_dir, "calib")
        calib = read_calib(calib_dir)
        idx_list = os.listdir(pc_dir)
        idx_list = [idx.split(".")[0] for idx in idx_list]
        idx_list.sort(key=int)
        # for item in all data
        with torch.no_grad():
            for idx in tqdm(idx_list):
                # getitem
                pc = read_pc_from_bin(os.path.join(pc_dir, idx+".bin"))
                img = read_image(os.path.join(img2_dir, idx+".png"))
                input_dict = {
                    "lidar": {
                        "type": "lidar",
                        "points": pc,
                    },
                    "metadata": {
                        "image_idx": int(idx),
                        "image_shape": None,
                    },
                    "calib": None,
                    "cam": {}
                }
                calib_dict = {
                    'rect': calib.R0_rect,
                    'Trv2c': calib.Tr_velo_to_cam,
                    'P2': np.concatenate([calib.P2, np.array([[0, 0, 0, 1]])], axis=0),
                }
                input_dict['calib'] = calib_dict
                example = val_data.dataset._prep_func(input_dict)
                if "image_idx" in input_dict["metadata"]:
                    example["metadata"] = input_dict["metadata"]
                if "anchors_mask" in example:
                    example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
                example = merge_second_batch([example])
                val_example = example_convert_to_torch(example, torch.float32)
                detection = net(val_example)
                detections += detection
        detections = val_data.dataset.convert_detection_to_kitti_annos(detections)
        # save results
        result_path_step = Path(vis_dir)
        with open(result_path_step / "result.pkl", 'wb') as f:
            pickle.dump(detections, f)

def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "importance"
    ]
    for k, v in example.items():
        if k in float_names:
            # slow when directly provide fp32 data with dtype=torch.half
            example_torch[k] = torch.tensor(
                v, dtype=torch.float32, device=device).to(dtype)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.uint8, device=device)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = torch.tensor(
                    v1, dtype=dtype, device=device).to(dtype)
            example_torch[k] = calib
        elif k == "num_voxels":
            example_torch[k] = torch.tensor(v)
        else:
            example_torch[k] = v
    return example_torch

def setup_logger(log_dir):
    logger = Logger()
    logger.global_dir = log_dir
    return logger

def train_one_epoch(net, dataloader, optimizer, evaluater):
    raise NotImplementedError

def validate(net, dataloader, evaluater):
    raise NotImplementedError

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SECOND')
    parser.add_argument('--tag',
                        type=str, metavar='TAG',
                        help='tag', default=None)
    parser.add_argument('--cfg',
                        type=str, metavar='CFG',
                        help='config file path')
    parser.add_argument('--is-val',
                        dest='is_val', 
                        default=False,
                        action='store_true', help='is validation dataset?')
    parser.add_argument('--rawdata-dir',
                        type=str, metavar='CFG',
                        help='config file path')
    parser.add_argument('--vis-dir',
                        type=str, metavar='CFG',
                        help='config file path')

    args = parser.parse_args()
    cfg = args.cfg
    tag = args.tag if args.tag is not None else f"SECOND-{time.time():.2f}"
    is_val = args.is_val
    rawdata_dir = args.rawdata_dir if not is_val else None
    vis_dir = args.vis_dir
    main(tag, cfg, is_val=is_val, rawdata_dir=rawdata_dir, vis_dir=vis_dir)