'''
File Created: Wednesday, 26th June 2019 3:15:12 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import unittest
import numpy as np
from PIL import Image
try:
    from ..dataloarder.augmentor import KittiAugmentor
    from ..dataloarder.kittidata import *
    from ..utils.utils import read_pc_from_bin
    from ..visualizer.vis import *
except:
    # Run script
    import sys
    sys.path.append('../')
    from det3.dataloarder.augmentor import KittiAugmentor
    from det3.dataloarder.kittidata import *
    from det3.utils.utils import read_pc_from_bin
    from det3.visualizer.vis import *

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
class TestKittiAugmentor(unittest.TestCase):
    def test_init(self):
        kitti_agmtor = KittiAugmentor()
        self.assertEqual(kitti_agmtor.dataset, "Kitti")
    
    def test_rot_obj(self):
        pc = read_pc_from_bin("./unit-test/data/test_KittiObj_000016.bin")
        calib = KittiCalib("./unit-test/data/test_KittiObj_000016.txt").read_calib_file()
        label = KittiLabel("./unit-test/data/test_KittiObj_000016_label.txt").read_label_file()

        bevimg = BEVImage(x_range=(0, 70), y_range=(-40, 40), grid_size=(0.05, 0.05))
        bevimg.from_lidar(pc, scale=1)
        for obj in label.data:
            bevimg.draw_box(obj, calib, bool_gt=True)
        bevimg_img = Image.fromarray(bevimg.data)
        bevimg_img.save(os.path.join('./unit-test/result/', 'test_KittiAugmentor_rotobj_origin.png'))

        kitti_agmtor = KittiAugmentor()
        label, pc = kitti_agmtor.rotate_obj(label, pc, calib, [-10/180.0 * np.pi, 10/180.0 * np.pi])
        bevimg = BEVImage(x_range=(0, 70), y_range=(-40, 40), grid_size=(0.05, 0.05))
        bevimg.from_lidar(pc, scale=1)
        for obj in label.data:
            bevimg.draw_box(obj, calib, bool_gt=True)
        bevimg_img = Image.fromarray(bevimg.data)
        bevimg_img.save(os.path.join('./unit-test/result/', 'test_KittiAugmentor_rotobj_result.png'))
        print(bcolors.WARNING + "Warning: TestKittiAugmentor:test_rot_obj: You should check the function manully.:)"+ bcolors.ENDC)
        print(bcolors.WARNING + os.path.join('Warning: TestKittiAugmentor:test_rot_obj:   ./unit-test/result/', 'test_KittiAugmentor_rotobj_origin.png') + bcolors.ENDC)
        print(bcolors.WARNING + os.path.join('Warning: TestKittiAugmentor:test_rot_obj:   ./unit-test/result/', 'test_KittiAugmentor_rotobj_result.png') + bcolors.ENDC)

    def test_tr_obj(self):
        pc = read_pc_from_bin("./unit-test/data/test_KittiObj_000016.bin")
        calib = KittiCalib("./unit-test/data/test_KittiObj_000016.txt").read_calib_file()
        label = KittiLabel("./unit-test/data/test_KittiObj_000016_label.txt").read_label_file()

        bevimg = BEVImage(x_range=(0, 70), y_range=(-40, 40), grid_size=(0.05, 0.05))
        bevimg.from_lidar(pc, scale=1)
        for obj in label.data:
            bevimg.draw_box(obj, calib, bool_gt=True)
        bevimg_img = Image.fromarray(bevimg.data)
        bevimg_img.save(os.path.join('./unit-test/result/', 'test_KittiAugmentor_trobj_origin.png'))

        kitti_agmtor = KittiAugmentor()
        label, pc = kitti_agmtor.tr_obj(label, pc, calib, dx_range = [-0.25, 0.25], dy_range = [-0.25, 0.25], dz_range = [-0.1, 0.1])
        bevimg = BEVImage(x_range=(0, 70), y_range=(-40, 40), grid_size=(0.05, 0.05))
        bevimg.from_lidar(pc, scale=1)
        for obj in label.data:
            bevimg.draw_box(obj, calib, bool_gt=True)
        bevimg_img = Image.fromarray(bevimg.data)
        bevimg_img.save(os.path.join('./unit-test/result/', 'test_KittiAugmentor_trobj_result.png'))
        print(bcolors.WARNING + "Warning: TestKittiAugmentor:test_tr_obj: You should check the function manully.:)"+ bcolors.ENDC)
        print(bcolors.WARNING + os.path.join('Warning: TestKittiAugmentor:test_tr_obj:   ./unit-test/result/', 'test_KittiAugmentor_trobj_origin.png') + bcolors.ENDC)
        print(bcolors.WARNING + os.path.join('Warning: TestKittiAugmentor:test_tr_obj:   ./unit-test/result/', 'test_KittiAugmentor_trobj_result.png') + bcolors.ENDC)

    def test_flip_pc(self):
        pc = read_pc_from_bin("./unit-test/data/test_KittiObj_000016.bin")
        calib = KittiCalib("./unit-test/data/test_KittiObj_000016.txt").read_calib_file()
        label = KittiLabel("./unit-test/data/test_KittiObj_000016_label.txt").read_label_file()

        bevimg = BEVImage(x_range=(0, 70), y_range=(-40, 40), grid_size=(0.05, 0.05))
        bevimg.from_lidar(pc, scale=1)
        for obj in label.data:
            bevimg.draw_box(obj, calib, bool_gt=True)
        bevimg_img = Image.fromarray(bevimg.data)
        bevimg_img.save(os.path.join('./unit-test/result/', 'test_KittiAugmentor_flippc_origin.png'))

        kitti_agmtor = KittiAugmentor()
        label, pc = kitti_agmtor.flip_pc(label, pc, calib)
        bevimg = BEVImage(x_range=(0, 70), y_range=(-40, 40), grid_size=(0.05, 0.05))
        bevimg.from_lidar(pc, scale=1)
        for obj in label.data:
            bevimg.draw_box(obj, calib, bool_gt=True)
        bevimg_img = Image.fromarray(bevimg.data)
        bevimg_img.save(os.path.join('./unit-test/result/', 'test_KittiAugmentor_flippc_result.png'))
        print(bcolors.WARNING + "Warning: TestKittiAugmentor:test_flip_pc: You should check the function manully.:)"+ bcolors.ENDC)
        print(bcolors.WARNING + os.path.join('Warning: TestKittiAugmentor:test_flip_pc:   ./unit-test/result/', 'test_KittiAugmentor_flippc_origin.png') + bcolors.ENDC)
        print(bcolors.WARNING + os.path.join('Warning: TestKittiAugmentor:test_flip_pc:   ./unit-test/result/', 'test_KittiAugmentor_flippc_result.png') + bcolors.ENDC)

if __name__ == "__main__":
    unittest.main()