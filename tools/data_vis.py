'''
File Created: Sunday, 24th March 2019 8:08:05 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
Usage: python3 tools/data_vis.py \
    --data-dir /usr/app/data/KITTI/dev \
    --idx-file /usr/app/data/KITTI/split_index/dev.txt \
    --output-dir /usr/app/vis/dev
'''
import argparse
import os
import sys
sys.path.append("../")
from PIL import Image
from det3.utils.utils import get_idx_list
from det3.dataloarder.data import KittiData
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
    args = parser.parse_args()
    data_dir = args.data_dir
    idx_path = args.idx_file
    output_dir = args.output_dir
    idx_list = get_idx_list(idx_path)
    for idx in idx_list:
        calib, img, label, pc = KittiData(data_dir, idx).read_data()
        bevimg = BEVImage(x_range=(0, 70), y_range=(-40, 40), grid_size=(0.05, 0.05))
        bevimg.from_lidar(pc, scale=1)
        for obj in label.read_kitti_label_file().data:
            bevimg.draw_box(obj, calib, bool_gt=True)
        bevimg_img = Image.fromarray(bevimg.data)
        bevimg_img.save(os.path.join(output_dir, idx+'.png'))

if __name__ == '__main__':
    main()
