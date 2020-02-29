'''
 File Created: Sat Feb 29 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
'''
import unittest
import numpy as np

class UTIO(unittest.TestCase):
    def test_read_txt(self):
        from det3.ops import read_txt
        path = "./unit-test/data/test_CarlaCalib_000000.txt"
        est = read_txt(path)
        gt = ["Tr_imu_to_velo_top: 1 0 0 0 0 1 0 0 0 0 1 -1.5 0 0 0 1"] + \
             ["Tr_imu_to_velo_left: 1 0 0 -2 0 -1 0 1 0 0 1 1 0 0 0 1"] + \
             ["Tr_imu_to_velo_right: 1 0 0 -2 0 -1 0 -1 0 0 1 1 0 0 0 1"]
        self.assertTrue(len(est) == len(gt))
        for est_, gt_ in zip(est, gt):
            self.assertTrue(est_ == gt_)

    def test_write_txt(self):
        from det3.ops import write_txt, read_txt
        data = ["Tr_imu_to_velo_top: 1 0 0 0 0 1 0 0 0 0 1 -1.5 0 0 0 1"] + \
               ["Tr_imu_to_velo_left: 1 0 0 -2 0 -1 0 1 0 0 1 1 0 0 0 1"] + \
               ["Tr_imu_to_velo_right: 1 0 0 -2 0 -1 0 -1 0 0 1 1 0 0 0 1"]
        path = "./unit-test/result/test_write_txt.txt"
        write_txt(data, path)
        est = read_txt(path)
        gt = read_txt("./unit-test/data/test_CarlaCalib_000000.txt")
        self.assertTrue(len(est) == len(gt))
        for est_, gt_ in zip(est, gt):
            self.assertTrue(est_ == gt_)

    def test_read_npy(self):
        from det3.ops import read_npy, write_npy
        gt = np.random.randn(1000).astype(np.float32)
        np.save("./unit-test/result/test_read_npy.npy", gt)
        est = read_npy("./unit-test/result/test_read_npy.npy")
        self.assertTrue(np.array_equal(gt, est))

    def test_write_npy(self):
        from det3.ops import read_npy, write_npy
        data = np.random.randn(1000).astype(np.float32)
        path = "./unit-test/result/test_write_npy.npy"
        write_npy(data, path)
        est = read_npy(path)
        gt = data
        self.assertTrue(np.array_equal(gt, est))

    def test_pcd_io(self):
        from det3.ops import write_pcd, read_pcd
        for dtype in [np.float32, np.float16]:
            data = np.random.randn(3*100).reshape(100, -1).astype(dtype)
            path = "./unit-test/result/test_pcd_io.pcd"
            write_pcd(data, path)
            est = read_pcd(path)
            self.assertTrue(np.array_equal(data, est))

    def test_bin_io(self):
        from det3.ops import write_bin, read_bin
        for dtype in [np.float32, np.float16]:
            data = np.random.randn(3*100).reshape(100, -1).astype(dtype)
            path = "./unit-test/result/test_pcd_io.bin"
            write_bin(data, path)
            est = read_bin(path, dtype).reshape(100, -1)
            self.assertTrue(np.array_equal(data, est))

    def test_pkl_io(self):
        from det3.ops import write_pkl, read_pkl
        for dtype in [np.float32, np.float16]:
            data = np.random.randn(3*100).reshape(100, -1).astype(dtype)
            path = "./unit-test/result/test_pcd_io.pkl"
            write_pkl(data, path)
            est = read_pkl(path).reshape(100, -1)
            self.assertTrue(np.array_equal(data, est))

    def test_img_io(self):
        from det3.ops import read_img, write_img
        read_data = read_img("./unit-test/data/det3_v0.1_arch.png")
        path = "./unit-test/result/test_img_io.png"
        write_img(read_data, path)
        write_data = read_img(path)
        self.assertTrue(np.array_equal(read_data, write_data))

if __name__ == "__main__":
    unittest.main()