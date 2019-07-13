'''
File Created: Wednesday, 1st May 2019 3:00:31 pm
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
import math
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw
import numpy as np
from tensorboardX import SummaryWriter
from det3.methods.voxelnet.config import cfg
from det3.methods.voxelnet.model import VoxelNet
from det3.methods.voxelnet.criteria import VoxelNetLoss
from det3.visualizer.vis import BEVImage
from det3.methods.voxelnet.utils import parse_grid_to_label
from det3.utils.utils import write_str_to_file
import kitti_common as kitti
from eval import get_official_eval_result, get_coco_eval_result

root_dir = __file__.split('/')
root_dir = os.path.join(root_dir[0], root_dir[1])
save_dir = os.path.join(root_dir, 'saved_weights', cfg.TAG)
log_dir = os.path.join(root_dir, 'logs', cfg.TAG)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
shutil.copy(os.path.join(root_dir, 'config.py'), os.path.join(log_dir, 'config.py'))
logging.basicConfig(filename=os.path.join(log_dir, 'log.txt'), level=logging.INFO)
tsbd = SummaryWriter(log_dir)

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
    best_loss1 = math.inf
    model = VoxelNet(in_channels=7,
                     out_gridsize=cfg.MIDGRID_SHAPE, bool_sparse=cfg.sparse)
    if cfg.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
        output_log("Use GPU: {} for training".format(cfg.gpu))
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function and optimizer
    criterion = VoxelNetLoss(cfg.alpha, cfg.beta, cfg.eta, cfg.gamma, cfg.lambda_rot)
    # optimizer = torch.optim.SGD(model.parameters(), cfg.lr,
    #                             momentum=cfg.momentum,
    #                             weight_decay=cfg.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_dict["lr"],
                                 betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=cfg.weight_decay, amsgrad=False)

    # optionally resume from a checkpoint
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            output_log("=> loading checkpoint '{}'".format(cfg.resume))
            checkpoint = torch.load(cfg.resume)
            cfg.start_epoch = checkpoint['epoch']
            best_loss1 = checkpoint['best_loss1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            output_log("=> loaded checkpoint '{}' (epoch {})"
                       .format(cfg.resume, checkpoint['epoch']))
        else:
            output_log("=> no checkpoint found at '{}'".format(cfg.resume))

    cudnn.benchmark = True
    if "KITTI" in cfg.DATADIR.split("/"):
        from det3.methods.voxelnet.kittidata import KittiDatasetVoxelNet, KITTIDataVoxelNet
        kitti_data = KITTIDataVoxelNet(data_dir=cfg.DATADIR, cfg=cfg, batch_size=cfg.batch_size).kitti_loaders
        train_loader = kitti_data['train']
        val_loader = KittiDatasetVoxelNet(data_dir=cfg.DATADIR, train_val_flag='val', cfg=cfg)
        val_loader_dev = KittiDatasetVoxelNet(data_dir=cfg.DATADIR, train_val_flag='dev', cfg=cfg)
    elif "CARLA" in cfg.DATADIR.split("/"):
        from det3.methods.voxelnet.carladata import CarlaDatasetVoxelNet, CarlaDataVoxelNet
        carla_data = CarlaDataVoxelNet(data_dir=cfg.DATADIR, cfg=cfg, batch_size=cfg.batch_size).carla_loaders
        train_loader = carla_data['train']
        val_loader = CarlaDatasetVoxelNet(data_dir=cfg.DATADIR, train_val_flag='val', cfg=cfg)
    else:
        raise NotImplementedError

    for epoch in range(cfg.start_epoch, cfg.epochs):
        adjust_learning_rate(optimizer, epoch, cfg.lr_dict)
        train_loss = train(train_loader, model, criterion, optimizer, epoch, cfg)
        tsbd.add_scalar('train/loss', train_loss, epoch)
        if (epoch != 0 and epoch % cfg.val_freq == 0) or epoch == cfg.epochs-1:
            val_loss_dev, val_ap_dict_dev = validate(val_loader_dev, model, criterion, epoch, cfg)
            tsbd.add_scalar('dev/loss', val_loss_dev, epoch)
            tsbd.add_scalar('dev/ap_bev_mod_0.7_0.7_0.7', val_ap_dict_dev["bev_AP_0"][1], epoch)
            tsbd.add_scalar('dev/ap_3d_mod_0.7_0.7_0.7', val_ap_dict_dev["3d_AP_0"][1], epoch)
            tsbd.add_scalar('dev/ap_bev_mod_0.7_0.5_0.5', val_ap_dict_dev["bev_AP_1"][1], epoch)
            tsbd.add_scalar('dev/ap_3d_mod_0.7_0.5_0.5', val_ap_dict_dev["3d_AP_1"][1], epoch)
            val_loss, val_ap_dict = validate(val_loader, model, criterion, epoch, cfg)
            tsbd.add_scalar('val/loss', val_loss, epoch)
            tsbd.add_scalar('val/ap_bev_mod_0.7_0.7_0.7', val_ap_dict["bev_AP_0"][1], epoch)
            tsbd.add_scalar('val/ap_3d_mod_0.7_0.7_0.7', val_ap_dict["3d_AP_0"][1], epoch)
            tsbd.add_scalar('val/ap_bev_mod_0.7_0.5_0.5', val_ap_dict["bev_AP_1"][1], epoch)
            tsbd.add_scalar('val/ap_3d_mod_0.7_0.5_0.5', val_ap_dict["3d_AP_1"][1], epoch)
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
    cls_losses = AverageMeter()
    reg_losses = AverageMeter()
    cls_pos_losses = AverageMeter()
    cls_neg_losses = AverageMeter()
    rot_rgl_losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, (tag, voxel_feature, coordinate, gt_pos_map, gt_neg_map, gt_target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if cfg.gpu is not None:
            voxel_feature = voxel_feature.cuda(cfg.gpu, non_blocking=True)
            coordinate = coordinate.cuda(cfg.gpu, non_blocking=True)
            gt_pos_map = gt_pos_map.cuda(cfg.gpu, non_blocking=True)
            gt_neg_map = gt_neg_map.cuda(cfg.gpu, non_blocking=True)
            gt_target = gt_target.cuda(cfg.gpu, non_blocking=True)

        # compute output
        est_pmap, est_rmap = model(voxel_feature, coordinate, batch_size=cfg.batch_size)
        output = {"obj":est_pmap, 'reg':est_rmap}
        target = {"obj":gt_pos_map, 'reg':gt_target, "neg-obj":gt_neg_map}
        loss_dict = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss_dict["loss"].item(), voxel_feature.size(0))
        cls_losses.update(loss_dict["cls_loss"].item(), voxel_feature.size(0))
        reg_losses.update(loss_dict["reg_loss"].item(), voxel_feature.size(0))
        cls_pos_losses.update(loss_dict["cls_pos_loss"].item(), voxel_feature.size(0))
        cls_neg_losses.update(loss_dict["cls_neg_loss"].item(), voxel_feature.size(0))
        rot_rgl_losses.update(loss_dict["rot_rgl_loss"].item(), voxel_feature.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_dict["loss"].backward()
        optimizer.step()

        # measure elapsed timeFalse
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.print_freq == 0:
            output_log('Epoch: [{0}][{1}/{2}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                       'cls loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                       'reg loss {reg_loss.val:.4f} ({reg_loss.avg:.4f})\t'
                       'cls pos loss {cls_pos_loss.val:.4f} ({cls_pos_loss.avg:.4f})\t'
                       'cls neg loss {cls_neg_loss.val:.4f} ({cls_neg_loss.avg:.4f})\t'
                       'rot rgl loss {rot_rgl_loss.val:.4f} ({rot_rgl_loss.avg:.4f})\t'.format(
                           epoch, i, len(train_loader), batch_time=batch_time,
                           data_time=data_time, loss=losses,
                           cls_loss=cls_losses, reg_loss=reg_losses,
                           cls_pos_loss=cls_pos_losses, cls_neg_loss=cls_neg_losses,
                           rot_rgl_loss=rot_rgl_losses))
    return losses.avg

def validate(val_loader, model, criterion, epoch, cfg):
    batch_time = AverageMeter()
    losses = AverageMeter()
    val_dir_data = os.path.join(log_dir, 'vals', str(epoch), "data_{}".format(val_loader.train_val_flag))
    val_dir_img = os.path.join(log_dir, 'vals', str(epoch), "img_{}".format(val_loader.train_val_flag))
    if os.path.isdir(val_dir_data) or os.path.isdir(val_dir_img):
        shutil.rmtree(val_dir_data, ignore_errors=True)
        shutil.rmtree(val_dir_img, ignore_errors=True)
    # switch to evaluate mode
    model.eval()
    os.makedirs(os.path.join(log_dir, 'vals'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'vals', str(epoch)), exist_ok=True)
    os.makedirs(val_dir_data, exist_ok=False)
    os.makedirs(val_dir_img, exist_ok=False)

    with torch.no_grad():
        end = time.time()
        visnum = 0
        for i, (tag, voxel_feature, coordinate, gt_pos_map, gt_neg_map, gt_target, anchors, pc, label, calib) in enumerate(val_loader):
            if cfg.gpu is not None:
                voxel_feature = torch.from_numpy(voxel_feature).contiguous().cuda(cfg.gpu, non_blocking=True)
                coordinate = torch.from_numpy(coordinate).contiguous().cuda(cfg.gpu, non_blocking=True)
                gt_pos_map = torch.from_numpy(gt_pos_map).contiguous().cuda(cfg.gpu, non_blocking=True)
                gt_neg_map = torch.from_numpy(gt_neg_map).contiguous().cuda(cfg.gpu, non_blocking=True)
                gt_target = torch.from_numpy(gt_target).contiguous().cuda(cfg.gpu, non_blocking=True)
            # compute output
            est_pmap, est_rmap = model(voxel_feature, coordinate, batch_size=cfg.batch_size)
            output = {"obj":est_pmap, 'reg':est_rmap}
            target = {"obj":gt_pos_map, 'reg':gt_target, "neg-obj":gt_neg_map}
            loss_dict = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss_dict["loss"].item(), voxel_feature.size(0))

            est_pmap_np = est_pmap.cpu().numpy()
            est_rmap_np = est_rmap.cpu().numpy()
            rec_label = parse_grid_to_label(est_pmap_np[0], est_rmap_np[0], anchors,
                                            anchor_size=(cfg.ANCHOR_L, cfg.ANCHOR_W, cfg.ANCHOR_H),
                                            cls=cfg.cls, calib=calib,
                                            threshold_score=cfg.RPN_SCORE_THRESH,
                                            threshold_nms=cfg.RPN_NMS_THRESH)
            res_path = os.path.join(val_dir_data, '{:06d}.txt'.format(tag))
            write_str_to_file(str(rec_label), res_path)
            visnum += 1
            if visnum < cfg.val_max_visnum:
                bevimg = BEVImage(x_range=cfg.x_range, y_range=cfg.y_range, grid_size=(0.05, 0.05))
                bevimg.from_lidar(pc[:, :], scale=1)
                for obj in label.data:
                    bevimg.draw_box(obj, calib, bool_gt=True, width=3)
                for obj in rec_label.data:
                    bevimg.draw_box(obj, calib, bool_gt=False, width=2) # The latter bbox should be with a smaller width
                bevimg_img = Image.fromarray(bevimg.data)
                bevimg_img.save(os.path.join(val_dir_img, '{:06d}.png'.format(tag)))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % cfg.print_freq == 0:
                output_log('Test: [{0}/{1}]\t'
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                           'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                               i, len(val_loader), batch_time=batch_time, loss=losses))
        assert len(val_loader) > 50
        det_path = val_dir_data
        dt_annos = kitti.get_label_annos(det_path)
        gt_path = os.path.join(val_loader.label2_dir)
        val_image_ids = os.listdir(det_path)
        val_image_ids = [int(itm.split(".")[0]) for itm in val_image_ids]
        gt_annos = kitti.get_label_annos(gt_path, val_image_ids)
        val_ap_str = get_official_eval_result(gt_annos, dt_annos, 0)
        val_ap_dict = parse_eval_ap(val_ap_str)

    return losses.avg, val_ap_dict

def output_log(s):
    print(s)
    logging.critical(s)

def adjust_learning_rate(optimizer, epoch, lr_dict):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_dict["mode"] == "const":
        lr = lr_dict["lr"]
    elif lr_dict["mode"] == "decay":
        lr = lr_dict["lr"] * (lr_dict["factor"] ** (epoch // lr_dict["cycle"]))
    elif lr_dict["mode"] == "super-converge":
        min_lr, max_lr = lr_dict["lr_range"]
        cycle = lr_dict["cycle"]
        half1 = np.linspace(max_lr, min_lr, num=np.ceil(cycle/2.0).astype(int))
        half2 = np.linspace(min_lr, max_lr, num=np.ceil(cycle/2.0).astype(int))[1:-1]
        entire = np.concatenate([half1, half2])
        lr = entire[epoch % (cycle-1)]
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_dir, filename), os.path.join(save_dir, 'best.pth.tar'))

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

def parse_eval_ap(s):
    res_dict = dict()
    lines = s.split('\n')
    for line in lines[:4]:
        if line.startswith("bbox AP:"):
            line = line.split(":")[-1]
            res_dict["bbox_AP_0"] = [float(res) for res in line.split(',')[-3:]]
        if line.startswith("bev  AP:"):
            line = line.split(":")[-1]
            res_dict["bev_AP_0"] = [float(res) for res in line.split(',')[-3:]]
        if line.startswith("3d   AP:"):
            line = line.split(":")[-1]
            res_dict["3d_AP_0"] = [float(res) for res in line.split(',')[-3:]]
    for line in lines[-4:]:
        if line.startswith("bbox AP:"):
            line = line.split(":")[-1]
            res_dict["bbox_AP_1"] = [float(res) for res in line.split(',')[-3:]]
        if line.startswith("bev  AP:"):
            line = line.split(":")[-1]
            res_dict["bev_AP_1"] = [float(res) for res in line.split(',')[-3:]]
        if line.startswith("3d   AP:"):
            line = line.split(":")[-1]
            res_dict["3d_AP_1"] = [float(res) for res in line.split(',')[-3:]]
    return res_dict

if __name__ == "__main__":
    main()
