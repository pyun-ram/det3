'''
File Created: Saturday, 22nd June 2019 4:31:40 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
Function: Convert Carla label(imu frame) to Kitti label(left-cam frame) for evaluation with kitti official scripts.
python3 tools/label_carla2kitti.py \
    --input-dir /usr/app/data/CARLA/train/label_imu/ \
    --calib-dir /usr/app/data/CARLA/train/calib/ \
    --output-dir /usr/app/data/CARLA/train/label_cam/
'''
import argparse
import os
import numpy as np
from det3.dataloader.carladata import CarlaLabel, CarlaCalib, CarlaObj
from det3.dataloader.kittidata import KittiLabel
from det3.utils.utils import write_str_to_file
def validate(data_dir:str, idx:str, label_Fcam:KittiLabel, save_dir:str):
    '''
    To validate the function of this script
    inputs:
        data_dir: dir of Carla data
        idx: "000000.txt"
        label_Fcam: result after converting
        save_dir: dir for saving visulizaition figures
    Note: the following code is dangerous, and should not be used in other place.
    '''
    from det3.dataloader.carladata import CarlaData
    from det3.visualizer.vis import FVImage, BEVImage
    from det3.dataloader.kittidata import KittiObj, KittiLabel, KittiCalib
    from PIL import Image
    tag = idx.split(".")[0]
    pc, label, calib = CarlaData(data_dir, tag).read_data()
    bevimg = BEVImage(x_range=(-100, 100), y_range=(-50, 50), grid_size=(0.05, 0.05))
    bevimg.from_lidar(np.vstack([pc['velo_top']]), scale=1)
    fvimg = FVImage()
    fvimg.from_lidar(calib, np.vstack([pc['velo_top']]))
    for obj in label.read_label_file().data:
        if obj.type == 'Car':
            bevimg.draw_box(obj, calib, bool_gt=True)
            fvimg.draw_box(obj, calib, bool_gt=True)

    kittilabel = KittiLabel()
    kittilabel.data = []
    kitticalib = KittiCalib(None)
    kitticalib.P2 = np.array([[450, 0., 600, 0.],
                              [0., 450, 180, 0.],
                              [0., 0., 1, 0.]])
    kitticalib.R0_rect = np.eye(4)
    kitticalib.Tr_velo_to_cam = np.zeros((4, 4))
    kitticalib.Tr_velo_to_cam[0, 1] = -1
    kitticalib.Tr_velo_to_cam[1, 2] = -1
    kitticalib.Tr_velo_to_cam[2, 0] = 1
    kitticalib.Tr_velo_to_cam[3, 3] = 1

    kitticalib.data = []
    for obj in label_Fcam.data:
        kittilabel.data.append(KittiObj(str(obj)))
    for obj in kittilabel.data:
        if obj.type == 'Car':
            fvimg.draw_box(obj, kitticalib, bool_gt=False)
            bevimg.draw_box(obj, kitticalib, bool_gt=False)
    bevimg_img = Image.fromarray(bevimg.data)
    bevimg_img.save(os.path.join(save_dir, "{}_bev.png".format(tag)))
    fvimg_img = Image.fromarray(fvimg.data)
    fvimg_img.save(os.path.join(save_dir, "{}_fv.png".format(tag)))

def main():
    parser = argparse.ArgumentParser(description='Split Dataset according to a txt file.')
    parser.add_argument('--input-dir',
                        type=str, metavar='INPUT DIR',
                        help='input dir')
    parser.add_argument('--calib-dir',
                        type=str, metavar='CALIB DIR',
                        help='calib dir')
    parser.add_argument('--output-dir',
                        type=str, metavar='OUTPUT DIR',
                        help='output dir')
    args = parser.parse_args()
    input_dir = args.input_dir
    calib_dir = args.calib_dir
    output_dir = args.output_dir
    assert os.path.isdir(input_dir)
    assert os.path.isdir(calib_dir)
    assert os.path.isdir(output_dir)
    idx_list = os.listdir(input_dir)
    num = len(idx_list)
    assert num == len(os.listdir(calib_dir))

    for idx in idx_list:
        print(idx)
        label = CarlaLabel(os.path.join(input_dir, idx)).read_label_file()
        calib = CarlaCalib(os.path.join(calib_dir, idx)).read_calib_file()
        label_Fcam = CarlaLabel()
        label_Fcam.data = []
        for obj in label.data:
            obj_Fcam = CarlaObj(str(obj))
            cns_Fimu = obj.get_bbox3dcorners()
            cns_Fcam = calib.imu2cam(cns_Fimu)
            cns_Fcam2d = calib.cam2imgplane(cns_Fcam)
            minx = float(np.min(cns_Fcam2d[:, 0]))
            maxx = float(np.max(cns_Fcam2d[:, 0]))
            miny = float(np.min(cns_Fcam2d[:, 1]))
            maxy = float(np.max(cns_Fcam2d[:, 1]))
            obj_Fcam.score = obj.score
            obj_Fcam.x, obj_Fcam.y, obj_Fcam.z = calib.imu2cam(np.array([obj.x, obj.y, obj.z]).reshape(1, 3))[0]
            obj_Fcam.w, obj_Fcam.l, obj_Fcam.h = obj.l, obj.h, obj.w
            obj_Fcam.ry = -obj.ry
            obj_Fcam.bbox_l, obj_Fcam.bbox_t, obj_Fcam.bbox_r, obj_Fcam.bbox_b = minx, miny, maxx, maxy
            label_Fcam.data.append(obj_Fcam)
        # validate("/usr/app/data/CARLA/train/", idx, label_Fcam, "/usr/app/vis/train/")
        write_str_to_file(str(label_Fcam), os.path.join(output_dir, idx))

if __name__=="__main__":
    main()
