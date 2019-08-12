'''
File Created: Monday, 12th August 2019 1:19:00 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import random
import time
import warnings
import os
import shutil
import logging
import math
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from det3.utils.torch_utils import GradientLogger, ActivationLogger
from det3.methods.pointnet.model import PointNetCls, feature_transform_regularizer
from det3.methods.pointnet.config import cfg
from det3.methods.pointnet.dataset import ShapeNetDataset
from det3.methods.pointnet.criteria import PointNetClsLoss

root_dir = __file__.split('/')
root_dir = os.path.join(root_dir[0], root_dir[1])
log_dir = os.path.join(root_dir, 'logs', cfg.task_dict["TAG"])
os.makedirs(log_dir, exist_ok=True)
os.makedirs(os.path.join(log_dir, 'eval_results'), exist_ok=True)
os.makedirs(os.path.join(log_dir, 'eval_results', 'data'), exist_ok=True)
os.makedirs(os.path.join(log_dir, 'eval_results', 'imgs'), exist_ok=True)
shutil.copy(os.path.join(root_dir, 'config.py'), os.path.join(log_dir, 'test_config.py'))
logging.basicConfig(filename=os.path.join(log_dir, 'log.txt'), level=logging.INFO)

def main():
    if cfg.train_dict["SEED"] is not None:
        random.seed(cfg.train_dict["SEED"])
        torch.manual_seed(cfg.train_dict["SEED"])
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    best_loss1 = math.inf
    model = PointNetCls(k=cfg.model_dict["K"],
                        feature_transform=cfg.model_dict["FEATURE_TRANSFORM"])
    if cfg.task_dict["GPU"] is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
        output_log("Use GPU: {} for training".format(cfg.task_dict["GPU"]))
        torch.cuda.set_device(cfg.task_dict["GPU"])
        model = model.cuda(cfg.task_dict["GPU"])
    else:
        model = torch.nn.DataParallel(model).cuda()
    criterion = PointNetClsLoss()
    if cfg.train_dict["RESUME"]:
        if os.path.isfile(cfg.train_dict["RESUME"]):
            output_log("=> loading checkpoint '{}'".format(cfg.train_dict["RESUME"]))
            checkpoint = torch.load(cfg.train_dict["RESUME"])
            cfg.train_dict["START_EPOCH"] = checkpoint['epoch']
            best_loss1 = checkpoint['best_loss1']
            model.load_state_dict(checkpoint['state_dict'])
            output_log("=> loaded checkpoint '{}' (epoch {})"
                       .format(cfg.train_dict["RESUME"], checkpoint['epoch']))
        else:
            output_log("=> no checkpoint found at '{}'".format(cfg.train_dict["RESUME"]))
    cudnn.benchmark = True

    test_dataset = ShapeNetDataset(
        root=cfg.data_dict["DATA_DIR"],
        classification=True,
        split='test',
        npoints=cfg.data_dict["NUM_POINTS"],
        data_augmentation=False)

    testdataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg.train_dict["BATCH_SIZE"],
            shuffle=True,
            num_workers=cfg.train_dict["NUM_DATALOAD_WKERS"])

    val_loss, val_ap_dict = evaluate(testdataloader, model, criterion, cfg)


def evaluate(val_loader, model, criterion, cfg):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # switch to evaluate mode
    model.eval()
    total_correct = 0
    total_testset = 0
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(tqdm(val_loader)):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            if cfg.task_dict["GPU"] is not None:
                points = points.contiguous().cuda(cfg.task_dict["GPU"], non_blocking=True)
                target = target.contiguous().cuda(cfg.task_dict["GPU"], non_blocking=True)

            # compute output
            current_batch_size = points.shape[0]
            pred, _, _ = model(points)
            batch_time.update(time.time() - end)
            loss = criterion(pred, target)
            losses.update(loss, current_batch_size)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            total_correct += correct.item()
            total_testset += points.size()[0]
    output_log("Test: Epoch: {} Losses: {} Accuracy: {}".format(
               0, losses.avg, total_correct / float(total_testset)))
    return losses.avg, {"acc": total_correct / float(total_testset)}

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def output_log(s):
    print(s)
    logging.critical(s)

if __name__ == "__main__":
    main()