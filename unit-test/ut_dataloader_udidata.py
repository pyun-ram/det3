'''
 File Created: Thu Mar 19 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
'''
import time
import torch
import unittest
import numpy as np
from det3.dataloader.udidata import UdiFrame, UdiCalib, UdiObj, UdiLabel

class UTUDIFRAME(unittest.TestCase):
    def test(self):
        self.assertTrue(UdiFrame.all_frames() == ", ".join(UdiFrame.Frame._member_names_))
        frame0 = UdiFrame("BASE")
        frame1 = UdiFrame("BASE")
        frame2 = UdiFrame("LIDARTOP")
        self.assertTrue(frame0 == frame1)
        self.assertTrue(frame0 == "BASE")
        self.assertTrue(frame0 == UdiFrame.Frame["BASE"])
        self.assertTrue(not frame1 == frame2)

class UTUDICALIB(unittest.TestCase):
    def test(self):
        from numpy.linalg import inv
        from det3.ops import read_bin, hfill_pts
        pc_Fbase = read_bin("./unit-test/data/ut_UdiCalib_0_front.bin", dtype=np.float32).reshape(-1, 4)
        calib = UdiCalib("./unit-test/data/ut_UdiCalib_0.txt").read_calib_file()
        pc_Flfront = calib.transform(pc_Fbase[:, :3], UdiFrame("BASE"), UdiFrame("LIDARFRONT"))
        T = ("0.980388 -0.016089 0.19642 0.747812 "
            "0.017425 0.999835 -0.005075 -0.042543 "
           "-0.196306 0.008398 0.980507 -0.982992 "
            "0.0 0.0 0.0 1.0".split(" "))
        T = np.array(T).reshape(4, 4).astype(np.float32)
        org_pts_h = hfill_pts(pc_Fbase)
        pc_h = (inv(T).astype(np.float32) @ org_pts_h.T).T
        gt = pc_h[:, :3]
        self.assertTrue(np.array_equal(pc_Flfront, gt))
        pc_rcv = calib.transform(pc_Fbase[:, :3], UdiFrame("BASE"), UdiFrame("BASE"))
        self.assertTrue(np.array_equal(pc_Fbase[:, :3], pc_rcv))

class UTUDIOBJ(unittest.TestCase):
    def test_get_pts_idx(self):
        from det3.ops import crop_pts_3drot, read_npy
        from det3.dataloader.carladata import CarlaObj, CarlaObj, CarlaLabel, CarlaCalib
        pts = read_npy("./unit-test/data/test_CarlaAugmentor_000250.npy")
        calib = CarlaCalib("./unit-test/data/test_CarlaAugmentor_000250_calib.txt").read_calib_file()
        label = CarlaLabel("./unit-test/data/test_CarlaAugmentor_000250_label_imu.txt").read_label_file()
        pts = calib.lidar2imu(pts, key='Tr_imu_to_velo_top')
        for i in range(len(label)):
            obj = UdiObj(arr=np.array(label.bboxes3d[i, [3,4,5,2,1,0,6]]), cls="Car", score=0.99)
            idx = obj.get_pts_idx(pts)
            gt = label.data[i].get_pts_idx(pts)
            gt = np.where(gt)[0].flatten()
            self.assertTrue(set(idx.tolist()) == set(gt.tolist()))
        
    def test_corner_transformation(self):
        from det3.ops import crop_pts_3drot, read_npy
        from det3.dataloader.carladata import CarlaObj, CarlaObj, CarlaLabel, CarlaCalib
        pts = read_npy("./unit-test/data/test_CarlaAugmentor_000250.npy")
        calib = CarlaCalib("./unit-test/data/test_CarlaAugmentor_000250_calib.txt").read_calib_file()
        label = CarlaLabel("./unit-test/data/test_CarlaAugmentor_000250_label_imu.txt").read_label_file()
        pts = calib.lidar2imu(pts, key='Tr_imu_to_velo_top')
        for i in range(len(label)):
            obj = UdiObj(arr=np.array(label.bboxes3d[i, [3,4,5,2,1,0,6]]), cls="Car", score=0.99)
            cns = obj.get_bbox3d_corners()
            gt = label.data[i].get_bbox3dcorners()
            self.assertTrue(np.allclose(cns, gt, atol=1e-2))
            obj_cp = UdiObj()
            obj_cp.from_corners(cns, obj.cls, obj.score)
            self.assertTrue(obj_cp.equal(obj))

class UTUDILABLE(unittest.TestCase):
    def test_baselable(self):
        from det3.ops import read_bin, read_json
        label_path = f"./unit-test/data/ut_UdiLabel_21_bin.json"
        pc_path = f"./unit-test/data/ut_UdiLabel_21.bin"
        pc = read_bin(pc_path, dtype=np.float32)
        label = UdiLabel(label_path).read_label_file()
        json_ctt = read_json(label_path)
        self.assertTrue(len(label) == len(json_ctt["elem"]))
        label_sub = UdiLabel()
        for i, obj in enumerate(label):
            if i >= 5:
                break
            label_sub.add_obj(obj.copy())
        self.assertTrue(label_sub.equal(label))

    def test_by_visualization(self):
        from det3.ops import read_bin
        from det3.visualizer.vis import BEVImage, FVImage
        from tqdm import tqdm
        import numpy as np
        from det3.utils.utils import roty, rotz
        label_path = f"./unit-test/data/ut_UdiLabel_21_bin.json"
        pc_path = f"./unit-test/data/ut_UdiLabel_21.bin"
        calib_path = f"./unit-test/data/ut_UdiLabel_21.txt"
        pc = read_bin(pc_path, dtype=np.float32).reshape(-1, 4)
        label = UdiLabel(label_path).read_label_file()
        calib = UdiCalib(calib_path).read_calib_file()
        bev_img = BEVImage(x_range=(-100, 100), y_range=(-50, 50), grid_size=(0.05, 0.05))
        bev_img.from_lidar(pc, scale=1)
        for obj in label:
            bev_img.draw_box(obj, None, bool_gt=True, text=obj.cls[0])
        bev_img.save(f"./unit-test/result/test_UdiLabel_21.png")
        fv_img = FVImage()
        vcam_T = np.eye(4)
        vcam_T[:3, 3] =  np.array([-3, 0, 3])
        vcam_T[:3, :3] = rotz(0) @ roty(np.pi*0.1)
        calib.vcam_T = vcam_T
        fv_img.from_lidar(calib, pc, scale=2)
        for obj in label:
            fv_img.draw_3dbox(obj, calib, bool_gt=True)
        fv_img.save(f"./unit-test/result/test_UdiLabel_21_fv.png")
        print("Please manually check ./unit-test/result/test_UdiLabel_21.png")

if __name__ == "__main__":
    unittest.main()
