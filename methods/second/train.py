'''
File Created: Monday, 7th October 2019 8:35:49 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import argparse
from det3.methods.second.utils import Logger
from det3.methods.second.builder import (voxelizer_builder, box_coder_builder,
                                         second_builder, dataloader_builder,
                                         optimizer_builder, evaluater_builder,
                                         model_manager_builder)
from shutil import copy
import sys
def main(tag, cfg_path):
    # Initilization
        # initilize logger
    logger = Logger(path=None)
        # initilize folder structure
    log_dir = None
    saved_weights_dir = None
    cfg = load_config_file(path=cfg_path) # copy and load specifically the new copy
    # build net
    voxelizer = voxelizer_builder.build(voxelizer_cfg=cfg["voxelizer"])
    box_coder = box_coder_builder.build(box_coder_cfg=cfg["box_coder"])
    net = second_builder.build(net_cfg=cfg["net"], voxelizer=voxelizer, box_coder=box_coder)
    # build dataloader
    dataloader_cfg = cfg["data_loader"]
    train_dataloader = dataloader_builder.build(dataloader_cfg=dataloader_cfg["train"])
    val_dataloader = dataloader_builder.build(dataloader_cfg=dataloader_cfg["val"])
    # build optimizer
    optimizer = optimizer_builder.build(opt_cfg=cfg["optimizer"])
    # build evaluater
    evaluater = evaluater_builder.build(evaluater_cfg=cfg["evaluater"])
    # build model_manager
    model_manager = model_manager_builder.build(model_manager_cfg=cfg["weight_manager"])
    # load weight
    resume_weight(model_manager, net, weight_path=None)
    # train
    epoch=None
    for i in range(epoch):
        train_one_epoch(net, dataloader=train_dataloader,  optimizer=optimizer, evaluater=evaluater, logger=logger)
        validate(net, dataloader=val_dataloader, evaluater=evaluater, logger=logger)
        save_weight(model_manager, net, weight_path=None)

def load_config_file(path) -> dict:
    # backup config file
    # load config file (backup copy)
    # log config file (log & print)
    # return cfg (dict)
    raise NotImplementedError

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
    tag = args.tag if args.tag is not None else f"SECOND-{time.time():.2f}"
    cfg = args.cfg
    print(tag, cfg)
    sys.exit("DEBUG")
    main(tag, cfg)