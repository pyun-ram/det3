'''
File Created: Monday, 7th October 2019 8:35:49 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import os
import sys
import time
import argparse
import torch
import torch.utils.data
import numpy as np
from pathlib import Path
from shutil import copy
from det3.methods.second.utils import Logger, load_module
from det3.methods.second.builder import (voxelizer_builder, box_coder_builder,
                                         similarity_calculator_builder, 
                                         anchor_generator_builder, target_assigner_builder,
                                         second_builder, dataloader_builder,
                                         optimizer_builder, evaluater_builder,
                                         model_manager_builder)

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


def main(tag, cfg_path):
    root_dir = Path(__file__).parent
    log_dir = root_dir/"logs"/tag
    saved_weights_dir = root_dir/"saved_weights"/tag
    log_dir.mkdir(parents=True, exist_ok=True)
    saved_weights_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir)
    cfg = load_config_file(cfg_path=cfg_path, log_dir=log_dir)

    # build net
    voxelizer = voxelizer_builder.build(voxelizer_cfg=cfg.Voxelizer)
    anchor_generator = anchor_generator_builder.build(anchor_generator_cfg=cfg.AnchorGenerator)
    box_coder = box_coder_builder.build(box_coder_cfg=cfg.BoxCoder)
    similarity_calculator = similarity_calculator_builder.build(similarity_calculator_cfg=cfg.SimilarityCalculator)
    target_assigner = target_assigner_builder.build(target_assigner_cfg=cfg.TargetAssigner,
                                                    box_coder=box_coder,
                                                    anchor_generators=[anchor_generator],
                                                    region_similarity_calculators=[similarity_calculator])
    net = second_builder.build(cfg=cfg.Net, voxelizer=voxelizer, target_assigner=target_assigner)
    # build dataloader
    train_data = dataloader_builder.build(cfg.Net, cfg.TrainDataLoader,
                                          voxelizer, target_assigner, training=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=cfg.TrainDataLoader["batch_size"],
        shuffle=True,
        num_workers=cfg.TrainDataLoader["num_workers"],
        pin_memory=False,
        collate_fn=merge_second_batch,
        worker_init_fn=_worker_init_fn,
        drop_last=True)
    val_data = dataloader_builder.build(cfg.Net, cfg.ValDataLoader,
                                          voxelizer, target_assigner, training=False)
    val_dataloader = torch.utils.data.DataLoader(
        val_data,
        batch_size=cfg.TrainDataLoader["batch_size"],
        shuffle=True,
        num_workers=cfg.TrainDataLoader["num_workers"],
        pin_memory=False,
        collate_fn=merge_second_batch,
        worker_init_fn=_worker_init_fn,
        drop_last=True)
    # build optimizer
    optimizer, lr_scheduler = optimizer_builder.build(optimizer_cfg=cfg.Optimizer, lr_scheduler_cfg=cfg.LRScheduler, net=net)
    # build evaluater
    # evaluater = evaluater_builder.build(evaluater_cfg=cfg["evaluater"])
    # build model_manager
    # model_manager = model_manager_builder.build(model_manager_cfg=cfg["weight_manager"])
    # load weight
    # resume_weight(model_manager, net, weight_path=None)
    # train
    epoch=None
    for i in range(epoch):
        train_one_epoch(net, dataloader=train_dataloader,  optimizer=optimizer, evaluater=evaluater, logger=logger)
        validate(net, dataloader=val_dataloader, evaluater=evaluater, logger=logger)
        save_weight(model_manager, net, weight_path=None)

def setup_logger(log_dir):
    logger = Logger()
    logger.global_dir = log_dir
    return logger

def load_config_file(cfg_path, log_dir) -> dict:
    assert os.path.isfile(cfg_path)
    bkup_path = Path(log_dir)/"config.py"
    copy(cfg_path, bkup_path)
    cfg = load_module(bkup_path, "cfg")
    return cfg

def resume_weight(model_manager, net, weight_path):
    raise NotImplementedError

def save_weight(model_manager, net, weight_path):
    raise NotImplementedError

def train_one_epoch(net, dataloader, optimizer, evaluater, logger):
    raise NotImplementedError

def validate(net, dataloader, evaluater, logger):
    raise NotImplementedError

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SECOND')
    parser.add_argument('--tag',
                        type=str, metavar='TAG',
                        help='tag', default=None)
    parser.add_argument('--cfg',
                        type=str, metavar='CFG',
                        help='config file path')
    args = parser.parse_args()
    cfg = args.cfg
    tag = args.tag if args.tag is not None else f"SECOND-{time.time():.2f}"
    main(tag, cfg)