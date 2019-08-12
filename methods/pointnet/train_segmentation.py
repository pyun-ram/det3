'''
File Created: Thursday, 8th August 2019 2:34:01 pm
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
from det3.methods.pointnet.model import PointNetDenseCls, feature_transform_regularizer
from det3.methods.pointnet.config import cfg
from det3.methods.pointnet.dataset import ShapeNetDataset
from det3.methods.pointnet.criteria import PointNetSegLoss

root_dir = __file__.split('/')
root_dir = os.path.join(root_dir[0], root_dir[1])
save_dir = os.path.join(root_dir, 'saved_weights', cfg.task_dict["TAG"])
log_dir = os.path.join(root_dir, 'logs', cfg.task_dict["TAG"])
os.makedirs(save_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(os.path.join(log_dir, "train_grad"), exist_ok=True)
os.makedirs(os.path.join(log_dir, "train_actv"), exist_ok=True)
shutil.copy(os.path.join(root_dir, 'config.py'), os.path.join(log_dir, 'config.py'))
logging.basicConfig(filename=os.path.join(log_dir, 'log.txt'), level=logging.INFO)
tsbd = SummaryWriter(log_dir)
grad_logger = GradientLogger()
actv_logger = ActivationLogger()

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

    dataset = ShapeNetDataset(
        root=cfg.data_dict["DATA_DIR"],
        classification=False,
        class_choice=[cfg.data_dict["CLASS_CHOICE"]])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train_dict["BATCH_SIZE"],
        shuffle=True,
        num_workers=cfg.train_dict["NUM_DATALOAD_WKERS"])
    test_dataset = ShapeNetDataset(
        root=cfg.data_dict["DATA_DIR"],
        classification=False,
        class_choice=[cfg.data_dict["CLASS_CHOICE"]],
        split='test',
        data_augmentation=False)
    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.train_dict["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg.train_dict["NUM_DATALOAD_WKERS"])
    model = PointNetDenseCls(k=dataset.num_seg_classes,
                        feature_transform=cfg.model_dict["FEATURE_TRANSFORM"])
    if cfg.train_dict["LOG_GRAD"]:
        grad_logger.set_model(model)
    if cfg.train_dict["LOG_ACTV"]:
        actv_logger.set_model(model)
    if cfg.task_dict["GPU"] is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
        output_log("Use GPU: {} for training".format(cfg.task_dict["GPU"]))
        torch.cuda.set_device(cfg.task_dict["GPU"])
        model = model.cuda(cfg.task_dict["GPU"])
    else:
        model = torch.nn.DataParallel(model).cuda()
    if cfg.train_dict["LR_DICT"]["TYPE"] == "decay":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_dict["LR_DICT"]["LR"], betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.train_dict["LR_DICT"]["STEP_SIZE"],
                                                    gamma=cfg.train_dict["LR_DICT"]["GAMMA"])
    else:
        raise NotImplementedError
    criterion = PointNetSegLoss()
    if cfg.train_dict["RESUME"]:
        if os.path.isfile(cfg.train_dict["RESUME"]):
            output_log("=> loading checkpoint '{}'".format(cfg.train_dict["RESUME"]))
            checkpoint = torch.load(cfg.train_dict["RESUME"])
            cfg.train_dict["START_EPOCH"] = checkpoint['epoch']
            best_loss1 = checkpoint['best_loss1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            output_log("=> loaded checkpoint '{}' (epoch {})"
                       .format(cfg.train_dict["RESUME"], checkpoint['epoch']))
        else:
            output_log("=> no checkpoint found at '{}'".format(cfg.train_dict["RESUME"]))
    cudnn.benchmark = True
    for epoch in range(cfg.train_dict["START_EPOCH"], cfg.train_dict["EPOCHS"]):
        train_loss, train_grad_dict, train_actv_dict = train(dataloader, model, criterion, optimizer, epoch, cfg)
        tsbd.add_scalar('train/loss', train_loss, epoch)
        grad_logger.save_pkl(train_grad_dict, os.path.join(log_dir, "train_grad", "{}.pkl".format(epoch)))
        grad_logger.plot(train_grad_dict, os.path.join(log_dir, "train_grad", "{}.png".format(epoch)))
        actv_logger.save_pkl(train_actv_dict, os.path.join(log_dir, "train_actv", "{}.pkl".format(epoch)))
        actv_logger.plot(train_actv_dict, os.path.join(log_dir, "train_actv", "{}.png".format(epoch)))
        tsbd.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)        
        if (epoch != 0 and epoch % cfg.train_dict["VAL_FREQ"] == 0) or epoch == cfg.train_dict["EPOCHS"]-1:
            val_loss, val_ap_dict = validate(testdataloader, model, criterion, epoch, cfg)
            tsbd.add_scalar('val/loss', val_loss, epoch)
            tsbd.add_scalar('val/acc', val_ap_dict["acc"], epoch)
            save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss1': best_loss1,
            'optimizer' : optimizer.state_dict(),
        }, False, save_dir=save_dir, filename=str(epoch)+'.pth.tar')

def train(train_loader, model, criterion, optimizer, epoch, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        points, target = data
        points = points.transpose(2, 1)
        target = target.reshape(-1, 1)[:, 0] - 1
        if cfg.task_dict["GPU"] is not None:
            points = points.contiguous().cuda(cfg.task_dict["GPU"], non_blocking=True)
            target = target.contiguous().cuda(cfg.task_dict["GPU"], non_blocking=True)

        current_batch_size = points.shape[0]
        pred, trans, trans_feat = model(points)
        pred = pred.view(-1, train_loader.dataset.num_seg_classes)
        loss = criterion(pred, target)
        if cfg.model_dict["FEATURE_TRANSFORM"]:
            loss += feature_transform_regularizer(trans_feat) * 0.001        
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        # measure accuracy and record loss
        losses.update(loss, current_batch_size)
        accuracies.update(correct.item()/float(current_batch_size * 2500), current_batch_size)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed timeFalse
        batch_time.update(time.time() - end)
        end = time.time()
        output_log('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {acc.val:.4f} ({acc.avg:.4f})\t'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, acc=accuracies))
    return losses.avg, grad_logger.log(epoch), actv_logger.log(epoch)

def validate(val_loader, model, criterion, epoch, cfg):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # switch to evaluate mode
    model.eval()
    shape_ious = []
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(tqdm(val_loader)):
            points, target = data
            points = points.transpose(2, 1)
            if cfg.task_dict["GPU"] is not None:
                points = points.contiguous().cuda(cfg.task_dict["GPU"], non_blocking=True)
                target = target.contiguous().cuda(cfg.task_dict["GPU"], non_blocking=True)

            # compute output
            current_batch_size = points.shape[0]
            pred, _, _ = model(points)
            pred = pred.view(-1, val_loader.dataset.num_seg_classes)
            target = target.view(-1, 1)[:, 0] - 1
            batch_time.update(time.time() - end)
            loss = criterion(pred, target)
            losses.update(loss, current_batch_size)

            pred_choice = pred.data.max(1)[1]
            pred_np = pred_choice.cpu().data.numpy()
            target_np = target.cpu().data.numpy()
            for shape_idx in range(target_np.shape[0]):
                parts = range(val_loader.dataset.num_seg_classes)#np.unique(target_np[shape_idx])
                part_ious = []
                for part in parts:
                    I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                    U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                    if U == 0:
                        iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
                    else:
                        iou = I / float(U)
                    part_ious.append(iou)
                shape_ious.append(np.mean(part_ious))
    output_log("mIOU for class {}: {}".format(cfg.data_dict["CLASS_CHOICE"], np.mean(shape_ious)))
    return losses.avg, {"acc": 0}

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_dir, filename), os.path.join(save_dir, 'best.pth.tar'))

def output_log(s):
    print(s)
    logging.critical(s)

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

if __name__ == "__main__":
    main()
