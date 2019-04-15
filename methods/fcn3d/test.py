'''
File Created: Wednesday, 27th March 2019 9:29:20 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
Note: The code skeleton is based on https://github.com/pytorch/examples/blob/master/imagenet/main.py
'''

import random
import time
import sys
sys.path.append('../')
import warnings
import os
import shutil
import logging
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from PIL import Image
from det3.methods.fcn3d.config import cfg
from det3.methods.fcn3d.model import FCN3D
from det3.methods.fcn3d.criteria import FCN3DLoss
from det3.methods.fcn3d.data import KittiDatasetFCN3D
from det3.methods.fcn3d.utils import parse_grid_to_label
from det3.visualizer.vis import BEVImage

root_dir = __file__.split('/')
root_dir = os.path.join(root_dir[0], root_dir[1])
log_dir = os.path.join(root_dir, 'logs', cfg.TAG)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(os.path.join(log_dir, 'eval_results'), exist_ok=True)
os.makedirs(os.path.join(log_dir, 'eval_results', 'data'), exist_ok=True)
os.makedirs(os.path.join(log_dir, 'eval_results', 'imgs'), exist_ok=True)
shutil.copy(os.path.join(root_dir, 'config.py'), os.path.join(log_dir, 'test_config.py'))
logging.basicConfig(filename=os.path.join(log_dir, 'log.txt'), level=logging.INFO)

def output_log(s):
    print(s)
    logging.critical(s)

def write_str_to_file(s, file_path):
    with open(file_path, 'w+') as f:
        f.write(s)

def main():
    if cfg.seed is not None:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    model = FCN3D()
    if cfg.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
        output_log("Use GPU: {} for training".format(cfg.gpu))
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function and optimizer
    criterion = FCN3DLoss(alpha=cfg.alpha, beta=cfg.beta, eta=cfg.eta, gamma=cfg.gamma)

    # optionally resume from a checkpoint
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            output_log("=> loading checkpoint '{}'".format(cfg.resume))
            checkpoint = torch.load(cfg.resume)
            cfg.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            output_log("=> loaded checkpoint '{}' (epoch {})"
                        .format(cfg.resume, checkpoint['epoch']))
        else:
            output_log("=> no checkpoint found at '{}'".format(cfg.resume))

    cudnn.benchmark = True

    val_loader = KittiDatasetFCN3D(data_dir='/usr/app/data/KITTI', train_val_flag='dev', cfg=cfg)
    val_loss = evaluate(val_loader, model, criterion, cfg)

def evaluate(data_loader, model, criterion, cfg):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (tag, voxel, gt_objgrid, gt_reggrid, pc, label, calib) in enumerate(data_loader):
            if cfg.gpu is not None:
                voxel = torch.from_numpy(voxel).cuda(cfg.gpu, non_blocking=True)
                gt_objgrid = torch.from_numpy(gt_objgrid).cuda(cfg.gpu, non_blocking=True)
                gt_reggrid = torch.from_numpy(gt_reggrid).cuda(cfg.gpu, non_blocking=True)

            # compute output
            est_objgrid, est_reggrid = model(voxel)
            output = {"obj":est_objgrid, 'reg':est_reggrid}
            target = {"obj":gt_objgrid, 'reg':gt_reggrid}
            loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.item(), voxel.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            bevimg = BEVImage(x_range=cfg.x_range, y_range=cfg.y_range, grid_size=(0.05, 0.05))
            bevimg.from_lidar(pc[:, :], scale=1)

            for obj in label.data:
                if obj.type in cfg.KITTI_cls[cfg.cls]:
                    bevimg.draw_box(obj, calib, bool_gt=True, width=3)

            rec_label = parse_grid_to_label(est_objgrid.cpu().numpy()[0, ::], est_reggrid.cpu().numpy()[0, ::],
                                            score_threshold=cfg.threshold, nms_threshold=cfg.nms_threshold,
                                            calib=calib, cls=cfg.cls,
                                            res=tuple([cfg.scale * _d for _d in cfg.resolution]),
                                            x_range=cfg.x_range,
                                            y_range=cfg.y_range,
                                            z_range=cfg.z_range)
            for obj in rec_label.data:
                if obj.type in cfg.KITTI_cls[cfg.cls]:
                    bevimg.draw_box(obj, calib, bool_gt=False, width=2) # The latter bbox should be with a smaller width

            bevimg_img = Image.fromarray(bevimg.data)
            bevimg_img.save(os.path.join(log_dir, 'eval_results', 'imgs', '{:06d}.png'.format(tag)))
            res_path = os.path.join(log_dir, 'eval_results', 'data', '{:06d}.txt'.format(tag))
            write_str_to_file(str(rec_label), res_path)
            output_log('write out {} objects to {:06d}'.format(len(str(rec_label).split('\n')), tag))
            if i % cfg.print_freq == 0:
                output_log('Test: [{0}/{1}]\t'
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                           'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                               i, len(data_loader), batch_time=batch_time, loss=losses))
    return losses.avg

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
