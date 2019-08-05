'''
File Created: Sunday, 24th March 2019 8:08:05 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
Usage: python3 tools/data_vis.py \
    --data-dir /usr/app/data/KITTI/dev \
    --idx-file /usr/app/data/KITTI/split_index/dev.txt \
    --dataset KITTI \
    --output-dir /usr/app/vis/dev
'''
import argparse
import os
from PIL import Image
from det3.utils.utils import get_idx_list
from det3.visualizer.vis import BEVImage

def main():
    '''
    visualize data
    '''
    parser = argparse.ArgumentParser(description='Visulize Dataset')
    parser.add_argument('--data-dir',
                        type=str, metavar='INPUT PATH',
                        help='dataset dir')
    parser.add_argument('--idx-file',
                        type=str, metavar='INDEX FILE PATH',
                        help='the txt file containing the indeces of the smapled data')
    parser.add_argument('--output-dir',
                        type=str, metavar='OUTPUT PATH',
                        help='output dir')
    parser.add_argument('--dataset',
                        type=str, metavar='DATASET',
                        help='KITTI' or 'CARLA')
    args = parser.parse_args()
    data_dir = args.data_dir
    idx_path = args.idx_file
    output_dir = args.output_dir
    dataset = args.dataset.upper()
    idx_list = get_idx_list(idx_path)
    for idx in idx_list:
        print(idx)
        if dataset == "KITTI":
            from det3.dataloarder.kittidata import KittiData
            calib, img, label, pc = KittiData(data_dir, idx).read_data()
        elif dataset == "CARLA":
            from det3.dataloarder.carladata import CarlaData
            import numpy as np
            pc_dict, label, calib = CarlaData(data_dir, idx).read_data()
            pc = np.vstack([calib.lidar2imu(v, key="Tr_imu_to_{}".format(k)) for k, v in pc_dict.items()])
        bevimg = BEVImage(x_range=(0, 70), y_range=(-40, 40), grid_size=(0.05, 0.05))
        bevimg.from_lidar(pc, scale=1)
        for obj in label.read_label_file().data:
            bevimg.draw_box(obj, calib, bool_gt=True)
        bevimg_img = Image.fromarray(bevimg.data)
        bevimg_img.save(os.path.join(output_dir, idx+'.png'))

if __name__ == '__main__':
    main()
