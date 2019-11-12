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
import pickle
from shutil import copy
from det3.methods.second.utils import Logger, load_module
from det3.methods.second.builder import (voxelizer_builder, box_coder_builder,
                                         similarity_calculator_builder, 
                                         anchor_generator_builder, target_assigner_builder,
                                         second_builder, dataloader_builder,
                                         optimizer_builder, evaluater_builder,
                                         model_manager_builder)
from det3.methods.second.core.model_manager import save_models, restore
from tqdm import tqdm

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
    net = second_builder.build(cfg=cfg.Net, voxelizer=voxelizer, target_assigner=target_assigner).cuda()
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
        drop_last=False)
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
    # build optimizer
    optimizer, lr_scheduler = optimizer_builder.build(optimizer_cfg=cfg.Optimizer, lr_scheduler_cfg=cfg.LRScheduler, net=net)
    # build evaluater
    # evaluater = evaluater_builder.build(evaluater_cfg=cfg["evaluater"])
    evaluater = None
    if cfg.WeightManager["restore"] is not None:
        restore(cfg.WeightManager["restore"], net)
    logger = Logger()
    start_step = net.get_global_step()
    total_step = cfg.Optimizer["steps"]
    disp_itv = cfg.Task["disp_itv"]
    save_itv = cfg.Task["save_itv"]
    optimizer.zero_grad()
    step_times = []
    step = start_step
    t = time.time()
    while step < total_step:
        for example in train_dataloader:
            lr_scheduler.step(net.get_global_step())
            example_torch = example_convert_to_torch(example, torch.float32)
            batch_size = example["anchors"].shape[0]
            ret_dict = net(example_torch)
            cls_preds = ret_dict["cls_preds"]
            loss = ret_dict["loss"].mean()
            cls_loss_reduced = ret_dict["cls_loss_reduced"].mean()
            loc_loss_reduced = ret_dict["loc_loss_reduced"].mean()
            cls_pos_loss = ret_dict["cls_pos_loss"].mean()
            cls_neg_loss = ret_dict["cls_neg_loss"].mean()
            loc_loss = ret_dict["loc_loss"]
            cls_loss = ret_dict["cls_loss"]
            cared = ret_dict["cared"]
            labels = example_torch["labels"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            optimizer.step()
            optimizer.zero_grad()
            net.update_global_step()
            step_time = (time.time() - t)
            step_times.append(step_time)
            t = time.time()
            num_pos = int((labels > 0)[0].float().sum().cpu().numpy())
            num_neg = int((labels == 0)[0].float().sum().cpu().numpy())
            if step % disp_itv == 0 and step != 0:
                print(step, f"loss: {loss}, cls_pos_loss: {cls_pos_loss}, cls_neg_loss: {cls_neg_loss}, loc_loss: {loc_loss.mean()}")
                logger.log_tsbd_scalor("train/loss", loss, net.get_global_step())
            if step % save_itv == 0 and step != 0:
                save_models(saved_weights_dir, [net, optimizer], net.get_global_step(), max_to_keep=float('inf'))
                net.eval()
                detections = []
                result_path_step = log_dir / f"step_{net.get_global_step()}"
                result_path_step.mkdir(parents=True, exist_ok=True)
                logger.log_txt("#################################"+str(step))
                logger.log_txt("# VAL" + str(step))
                logger.log_txt("#################################"+str(step))
                for val_example in tqdm(val_dataloader):
                    val_example = example_convert_to_torch(val_example, torch.float32)
                    detections += net(val_example)
                result_dict = val_data.dataset.evaluation(detections,
                    label_dir=os.path.join(val_data.dataset.root_path,"training", "label_2"),
                    output_dir=str(result_path_step))
                logger.log_metrics(result_dict["detail"], step)
                with open(result_path_step / "result.pkl", 'wb') as f:
                    pickle.dump(detections, f)
                net.train()
            step += 1

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

def load_config_file(cfg_path, log_dir, backup=True) -> dict:
    assert os.path.isfile(cfg_path)
    if backup:
        bkup_path = Path(log_dir)/"config.py"
        copy(cfg_path, bkup_path)
        cfg = load_module(bkup_path, "cfg")
    else:
        cfg = load_module(cfg_path, "cfg")
    return cfg

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
    args = parser.parse_args()
    cfg = args.cfg
    tag = args.tag if args.tag is not None else f"SECOND-{time.time():.2f}"
    main(tag, cfg)