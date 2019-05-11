'''
File Created: Friday, 10th May 2019 4:47:03 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
This code is a reproduction of Deep3D Box.
Only the 3D bbox recovering is implemented.
Note: python3 methods/deep3dbox/main.py \
    --tag test \
    --data-dir /usr/app/data/KITTI/dev/
'''
import argparse
import os
import sys
sys.path.append("../")
import logging
import numpy as np
from det3.dataloarder.kittidata import KittiData, KittiLabel, KittiObj
from det3.utils.utils import roty
from det3.methods.deep3dbox.utils import recover_loc_by_geometry, filter_label_range

root_dir = __file__.split('/')
root_dir = os.path.join(root_dir[0], root_dir[1])
def main():
    # argparser
    parser = argparse.ArgumentParser(description='Deep3D Box (location recovering)')
    parser.add_argument('--tag',
                    type=str, metavar='TAG',
                    help='TAG')
    parser.add_argument('--data-dir',
                    type=str, metavar='INPUT PATH',
                    help='dataset dir (label_2)')
    args = parser.parse_args()
    tag = args.tag
    data_dir = args.data_dir
    log_dir = os.path.join(root_dir, 'logs', tag)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'eval_results'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'eval_results', 'data'), exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, 'log.txt'), level=logging.INFO)

    idx_list = [itm.split('.')[0] for itm in os.listdir(os.path.join(data_dir, 'label_2'))]
    idx_list.sort()
    for idx in idx_list:
        print(idx)
        calib, _, gt_label, _ = KittiData(data_dir, idx).read_data()
        gt_label = filter_label_range(gt_label, calib, x_range=(0, 70.4), y_range=(-40, 40), z_range=(-3, 1))
        est_label = KittiLabel()
        est_label.data = []
        for obj in gt_label.data:
            K = calib.P2
            l, w, h = obj.l, obj.w, obj.h
            x, y, z = recover_loc_by_geometry(K, obj.ry, l, w, h,
                                              calib=calib,
                                              bbox2d=np.array([obj.bbox_l,
                                                               obj.bbox_r,
                                                               obj.bbox_t,
                                                               obj.bbox_b]))
            est_obj = KittiObj("{}".format(obj))
            est_obj.x = x
            est_obj.y = y
            est_obj.z = z
            est_obj.score = 1
            est_label.data.append(est_obj)
        # save result
        res_path = os.path.join(log_dir, 'eval_results', 'data', '{}.txt'.format(idx))
        write_str_to_file(str(est_label), res_path)

def write_str_to_file(s, file_path):
    with open(file_path, 'w+') as f:
        f.write(s)

if __name__ == "__main__":
    main()
