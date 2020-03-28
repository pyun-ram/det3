'''
File Created: Sunday, 24th March 2019 8:08:05 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
Usage: python3 tools/data_vis.py \
    --data-dir /usr/app/data/MLOD-V2/train \
    --idx-file /usr/app/data/MLOD-V2/split_index/train.txt \
    --dataset CARLA \
    --output-dir /usr/app/vis/train
'''
import argparse
import os
from PIL import Image
from det3.utils.utils import get_idx_list, rotz, roty
from det3.visualizer.vis import BEVImage, FVImage
from multiprocessing import Pool
from tqdm import tqdm

def vis_fn(idx):
    global dataset, data_dir, output_dir
    if dataset == "KITTI":
        from det3.dataloader.kittidata import KittiData
        calib, img, label, pc = KittiData(data_dir, idx).read_data()
    elif dataset == "CARLA":
        from det3.dataloader.carladata import CarlaData
        import numpy as np
        pc_dict, label, calib = CarlaData(data_dir, idx).read_data()
        pc = np.vstack([calib.lidar2imu(v, key="Tr_imu_to_{}".format(k)) for k, v in pc_dict.items()])
    elif dataset == "UDI":
        from det3.dataloader.udidata import UdiData, UdiFrame
        import numpy as np
        udidata = UdiData(data_dir, idx).read_data()
        calib = udidata["calib"]
        label = udidata["label"]
        pc_dict = udidata["lidar"]
        for k, v in pc_dict.items():
            pc_dict[k] = calib.transform(v[:, :3],
                source_frame=UdiFrame(UdiData.lidar_to_frame(k)),
                target_frame=UdiFrame("BASE"))
        pc_merge = np.vstack([v for k, v in pc_dict.items()])
        bevimg = BEVImage(x_range=(-70, 70), y_range=(-40, 40), grid_size=(0.05, 0.05))
        bevimg.from_lidar(pc_merge, scale=1)
        for obj in label:
            bevimg.draw_box(obj, calib, bool_gt=True)
        bevimg_img = Image.fromarray(bevimg.data)
        bevimg_img.save(os.path.join(output_dir, idx+'.png'))

        for k, v in pc_dict.items():
            bevimg = BEVImage(x_range=(-70, 70), y_range=(-40, 40), grid_size=(0.05, 0.05))
            bevimg.from_lidar(v, scale=1)
            for obj in label:
                bevimg.draw_box(obj, calib, bool_gt=True)
            bevimg_img = Image.fromarray(bevimg.data)
            bevimg_img.save(os.path.join(output_dir, idx+f'_{k}.png'))

        fv_img = FVImage()
        vcam_T = np.eye(4)
        vcam_T[:3, 3] =  np.array([-3, 0, 3])
        vcam_T[:3, :3] = rotz(0) @ roty(np.pi*0.1)
        calib.vcam_T = vcam_T
        fv_img.from_lidar(calib, pc_merge, scale=2)
        for obj in label:
            fv_img.draw_3dbox(obj, calib, bool_gt=True)
        fv_img.save(os.path.join(output_dir, idx+'_fv.png'))
        return

    bevimg = BEVImage(x_range=(0, 70), y_range=(-40, 40), grid_size=(0.05, 0.05))
    bevimg.from_lidar(pc, scale=1)
    for obj in label.read_label_file().data:
        bevimg.draw_box(obj, calib, bool_gt=True)
    bevimg_img = Image.fromarray(bevimg.data)
    bevimg_img.save(os.path.join(output_dir, idx+'.png'))

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
    global dataset, data_dir, output_dir
    data_dir = args.data_dir
    idx_path = args.idx_file
    output_dir = args.output_dir
    dataset = args.dataset.upper()
    idx_list = get_idx_list(idx_path)
    with Pool(8) as p:
        r = list(tqdm(p.imap(vis_fn, idx_list), total=len(idx_list)))

if __name__ == '__main__':
    main()
