'''
File Created: Wednesday, 23rd October 2019 8:15:28 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
python3 methods/second/vis_result.py \
    --is-val \
    --data-dir /usr/app/data/KITTI/training \
    --vis-dir /usr/app/vis/demo-val \
    --pkl-path=/usr/app/vis/demo-val/result.pkl

python3 methods/second/vis_result.py \
    --data-dir /usr/app/data/KITTI-RAW/2011_09_26/2011_09_26_drive_0015_sync \
    --vis-dir /usr/app/vis/2011_09_26_drive_0015_sync \
    --pkl-path=/usr/app/vis/2011_09_26_drive_0015_sync/result.pkl
'''

import pickle
from det3.dataloader.kittidata import KittiLabel, KittiCalib, KittiObj, KittiData
from det3.visualizer.vis import FVImage
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from det3.utils.utils import read_image, read_pc_from_bin
import argparse

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

def main(is_val, data_dir, vis_dir, result_path):
    dt_data = pickle.load(open(result_path, 'rb'))
    save_dir = vis_dir
    if is_val:
        for dt in tqdm(dt_data):
            idx = dt['metadata']['image_idx']
            idx = "{:06d}".format(idx)
            calib, img, label, pc = KittiData(root_dir=data_dir, idx=idx).read_data()
            label_est = KittiLabel()
            label_est.data = []
            num_dt = len(dt['name'])
            for i in range(num_dt):
                obj = KittiObj()
                obj.type = dt['name'][i]
                obj.truncated = dt['truncated'][i]
                obj.occluded = dt['occluded'][i]
                obj.alpha = dt['alpha'][i]
                obj.bbox_l, obj.bbox_t, obj.bbox_r, obj.bbox_b = dt['bbox'][i]
                obj.l, obj.h, obj.w = dt['dimensions'][i]
                obj.x, obj.y, obj.z = dt['location'][i]
                obj.ry = dt['rotation_y'][i]
                obj.score = dt['score'][i]
                label_est.data.append(obj)
            fvimg = FVImage()
            fvimg.from_image(img)
            for obj in label.data:
                if (obj.x > obj.z) or (-obj.x > obj.z):
                    continue
                if obj.type in ['Car', 'Van']:
                    fvimg.draw_3dbox(obj, calib, bool_gt=True, width=3)
            for obj in label_est.data:
                if (obj.x > obj.z) or (-obj.x > obj.z):
                    continue
                fvimg.draw_3dbox(obj, calib, bool_gt=False, width=2, c=(255, 255, int(255 * obj.score), 255))
            fvimg_img = Image.fromarray(fvimg.data)
            fvimg_img.save(os.path.join(save_dir, "fv_{}.png".format(idx)))
    else:
        pc_dir = os.path.join(data_dir, "velodyne_points", "data")
        img2_dir = os.path.join(data_dir, "image_02", "data")
        calib_dir = os.path.join(data_dir, "calib")
        calib = read_calib(calib_dir)
        for dt in tqdm(dt_data):
            idx = dt['metadata']['image_idx']
            idx = "{:010d}".format(idx)
            pc = read_pc_from_bin(os.path.join(pc_dir, idx+".bin"))
            img = read_image(os.path.join(img2_dir, idx+".png"))
            label_est = KittiLabel()
            label_est.data = []
            num_dt = len(dt['name'])
            for i in range(num_dt):
                obj = KittiObj()
                obj.type = dt['name'][i]
                obj.truncated = dt['truncated'][i]
                obj.occluded = dt['occluded'][i]
                obj.alpha = dt['alpha'][i]
                obj.bbox_l, obj.bbox_t, obj.bbox_r, obj.bbox_b = dt['bbox'][i]
                obj.l, obj.h, obj.w = dt['dimensions'][i]
                obj.x, obj.y, obj.z = dt['location'][i]
                obj.ry = dt['rotation_y'][i]
                obj.score = dt['score'][i]
                label_est.data.append(obj)
            fvimg = FVImage()
            fvimg.from_image(img)
            # for obj in label.data:
            #     if obj.type in ['Car', 'Van']:
            #         fvimg.draw_3dbox(obj, calib, bool_gt=True, width=3)
            for obj in label_est.data:
                if (obj.x > obj.z) or (-obj.x > obj.z):
                    continue
                fvimg.draw_3dbox(obj, calib, bool_gt=False, width=2, c=(255, 255, int(obj.score*255), 255))
            fvimg_img = Image.fromarray(fvimg.data)
            fvimg_img.save(os.path.join(save_dir, "fv_{}.png".format(idx)))
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SECOND')
    parser.add_argument('--is-val',
                        dest='is_val', 
                        default=False,
                        action='store_true', help='is validation dataset?')
    parser.add_argument('--data-dir',
                        type=str, metavar='CFG')
    parser.add_argument('--vis-dir',
                        type=str, metavar='CFG')
    parser.add_argument('--pkl-path',
                        type=str, metavar='CFG')

    args = parser.parse_args()
    is_val = args.is_val
    data_dir = args.data_dir
    vis_dir = args.vis_dir
    pkl_path = args.pkl_path
    main(is_val=is_val, data_dir=data_dir, vis_dir=vis_dir, result_path=pkl_path)