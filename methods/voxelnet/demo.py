'''
File Created: Monday, 22nd July 2019 10:18:24 am
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
python3 methods/voxelnet/demo.py \
    --data-dir /usr/app/data/KITTI-RAW/2011_09_26/2011_09_26_drive_0023_sync \
    --vis-dir /usr/app/vis/demo/2011_09_26_drive_0023_sync
'''
import argparse
import os
import random
import warnings
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
import time
import sys
sys.path.append("../")
from det3.utils.utils import read_pc_from_bin
from det3.utils.utils import read_image
from det3.methods.voxelnet.model import VoxelNet
from det3.methods.voxelnet.config import cfg
from det3.methods.voxelnet.utils import *
from det3.dataloarder.kittidata import KittiCalib
from det3.visualizer.vis import BEVImage, FVImage
import tqdm

def load_model_from_path(model_name: str, path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(model_name, path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return getattr(foo, model_name)

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

        
def main():
    parser = argparse.ArgumentParser(description='Demo of 3D object detection')
    parser.add_argument('--data-dir',
                        type=str, metavar='DATA DIR',
                        help='data-dir')
    parser.add_argument('--vis-dir',
                        type=str, metavar='VIS DIR',
                        help='vis-dir')
    args = parser.parse_args()
    data_dir = args.data_dir
    vis_dir = args.vis_dir
    os.makedirs(vis_dir, exist_ok=True)

    if cfg.seed is not None:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    # load model
    model = VoxelNet(in_channels=7,
                     out_gridsize=cfg.MIDGRID_SHAPE, bool_sparse=cfg.sparse)
    if cfg.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
        print("Use GPU: {} for training".format(cfg.gpu))
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            print("=> loading checkpoint '{}'".format(cfg.resume))
            checkpoint = torch.load(cfg.resume)
            cfg.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(cfg.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(cfg.resume))

    # load data
    pc_dir = os.path.join(data_dir, "velodyne_points", "data")
    img2_dir = os.path.join(data_dir, "image_02", "data")
    calib_dir = os.path.join(data_dir, "calib")
    calib = read_calib(calib_dir)
    idx_list = os.listdir(pc_dir)
    idx_list = [idx.split(".")[0] for idx in idx_list]
    idx_list.sort(key=int)
    model.eval()
    batch_time = AverageMeter()
    anchors = create_anchors(x_range=cfg.x_range,
                            y_range=cfg.y_range,
                            target_shape=(cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH),
                            anchor_z=cfg.ANCHOR_Z,
                            anchor_size=(cfg.ANCHOR_L, cfg.ANCHOR_W, cfg.ANCHOR_H))    
    with torch.no_grad():
        for idx in tqdm.tqdm(idx_list):
            pc = read_pc_from_bin(os.path.join(pc_dir, idx+".bin"))
            img = read_image(os.path.join(img2_dir, idx+".png"))
            t1 = time.time()
            pc = filter_camera_angle(pc)
            voxel_dict = voxelize_pc(pc, res=cfg.resolution,
                                    x_range=cfg.x_range,
                                    y_range=cfg.y_range,
                                    z_range=cfg.z_range,
                                    num_pts_in_vox=cfg.voxel_point_count)
            voxel_feature = voxel_dict["feature_buffer"].astype(np.float32)
            coordinate = voxel_dict["coordinate_buffer"].astype(np.int64)
            voxel_feature = np.expand_dims(voxel_feature, 0)
            coordinate = np.expand_dims(coordinate, 0)
            voxel_feature = torch.from_numpy(voxel_feature).contiguous().cuda(cfg.gpu, non_blocking=True)
            coordinate = torch.from_numpy(coordinate).contiguous().cuda(cfg.gpu, non_blocking=True)
            est_pmap, est_rmap = model(voxel_feature, coordinate, batch_size=cfg.batch_size)
            output = {"obj":est_pmap, 'reg':est_rmap}
            est_pmap_np = est_pmap.cpu().numpy()
            est_rmap_np = est_rmap.cpu().numpy()
            rec_label = parse_grid_to_label(est_pmap_np[0], est_rmap_np[0], anchors,
                                            anchor_size=(cfg.ANCHOR_L, cfg.ANCHOR_W, cfg.ANCHOR_H),
                                            cls=cfg.cls, calib=calib, threshold_score=cfg.RPN_SCORE_THRESH,
                                            threshold_nms=cfg.RPN_NMS_THRESH)
            batch_time.update(time.time() - t1)
            fvimg_img = FVImage()
            fvimg_img.from_image(img)
            fvimg_lidar = FVImage()
            fvimg_lidar.from_lidar(calib, pc)
            bevimg = BEVImage(x_range=cfg.x_range, y_range=cfg.y_range, grid_size=(0.05, 0.05))
            bevimg.from_lidar(pc, scale=1)
            for obj in rec_label.data:
                if (obj.x > obj.z) or (-obj.x > obj.z):
                    continue
                if obj.type in cfg.KITTI_cls[cfg.cls]:
                    bevimg.draw_box(obj, calib, bool_gt=False, width=3)
                    fvimg_img.draw_3dbox(obj, calib, bool_gt=False, width=3) # The latter bbox should be with a smaller width
                    fvimg_lidar.draw_3dbox(obj, calib, bool_gt=False, width=3) # The latter bbox should be with a smaller width
            bevimg_img = Image.fromarray(bevimg.data)
            fvimgimg_img = Image.fromarray(fvimg_img.data)
            fvimglidar_img = Image.fromarray(fvimg_lidar.data)
            bevimg_img.save(os.path.join(vis_dir, 'bv_{:06d}.png'.format(int(idx))))
            fvimgimg_img.save(os.path.join(vis_dir, 'fv1_{:06d}.png'.format(int(idx))))
            fvimglidar_img.save(os.path.join(vis_dir, 'fv2_{:06d}.png'.format(int(idx))))
            print("Time: {batch_time.val:.3f} ({batch_time.avg:.3f})".format(batch_time=batch_time))

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