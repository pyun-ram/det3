'''
 File Created: Mon Mar 02 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
'''
import time
import torch
import unittest
import numpy as np

class UTBOXOP(unittest.TestCase):
    def test_get_corner_box_2drot(self):
        print("test_get_corner_box_2drot")
        from det3.ops import get_corner_box_2drot
        from det3.dataloader.carladata import CarlaObj
        # np
        boxes = np.array([[6, 3, 2, 5, np.pi * 0.1], [3, 2, 4, 4.5, np.pi * 0.2]])
        boxes = [boxes + np.abs(np.random.randn(10).reshape(2, 5)) for i in range(15)]
        boxes = np.vstack(boxes)
        gt = []
        for box in boxes:
            obj = CarlaObj()
            obj.x, obj.y, obj.l, obj.w, obj.ry = box.flatten()
            obj.z, obj.h =0, 0
            gt.append(obj.get_bbox3dcorners()[:4, :2])
        gt = np.stack(gt, axis=0)
        for test_dtype in [np.float32, np.float64]:
            boxes = boxes.astype(test_dtype)
            est = get_corner_box_2drot(boxes)
            self.assertTrue(np.allclose(est, gt, atol=1e-2))
            times = 100
            t1 = time.time()
            for i in range(times):
                est = get_corner_box_2drot(boxes)
            t = (time.time()-t1) / times * 1000
            print(f"time is {t:.2f} ms {est.dtype}")
            # torch
            boxes_ts = torch.from_numpy(boxes)
            est_ts = get_corner_box_2drot(boxes_ts)
            self.assertTrue(np.allclose(est_ts.numpy(), gt, atol=1e-2))
            t1 = time.time()
            for i in range(times):
                est_ts = get_corner_box_2drot(boxes_ts)
            t = (time.time()-t1) / times * 1000
            print(f"time is {t:.2f} ms {est_ts.dtype}")
            # torch gpu
            boxes_tsgpu = boxes_ts.cuda()
            est_tsgpu = get_corner_box_2drot(boxes_tsgpu)
            self.assertTrue(np.allclose(est_tsgpu.cpu().numpy(), gt, atol=1e-2))
            times = 100
            torch.cuda.synchronize()
            t1 = time.time()
            for i in range(times):
                est_tsgpu = get_corner_box_2drot(boxes_tsgpu)
            torch.cuda.synchronize()
            t = (time.time()-t1) / times * 1000
            print(f"time is {t:.2f} ms {est_tsgpu.dtype}")

    def test_get_corner_box_3drot(self):
        print("test_get_corner_box_3drot")
        from det3.ops import get_corner_box_3drot
        from det3.dataloader.carladata import CarlaObj
        # np
        boxes = np.array([[6, 3, 0, 2, 5, 1, np.pi * 0.1], [3, 2, 1, 4, 4.5, 2, np.pi * 0.2]])
        boxes = [boxes + np.abs(np.random.randn(14).reshape(2, 7)) for i in range(15)]
        boxes = np.vstack(boxes)
        gt = []
        for box in boxes:
            obj = CarlaObj()
            obj.x, obj.y, obj.z, obj.l, obj.w, obj.h, obj.ry = box.flatten()
            gt.append(obj.get_bbox3dcorners())
        gt = np.stack(gt, axis=0)
        for test_dtype in [np.float32, np.float64]:
            boxes = boxes.astype(test_dtype)
            est = get_corner_box_3drot(boxes)
            self.assertTrue(np.allclose(est, gt, atol=1e-2))
            times = 100
            t1 = time.time()
            for i in range(times):
                est = get_corner_box_3drot(boxes)
            t = (time.time()-t1) / times * 1000
            print(f"time is {t:.2f} ms {est.dtype}")
            # torch
            boxes_ts = torch.from_numpy(boxes)
            est_ts = get_corner_box_3drot(boxes_ts)
            self.assertTrue(np.allclose(est_ts.numpy(), gt, atol=1e-2))
            t1 = time.time()
            for i in range(times):
                est_ts = get_corner_box_3drot(boxes_ts)
            t = (time.time()-t1) / times * 1000
            print(f"time is {t:.2f} ms {est_ts.dtype}")
            # torch gpu
            boxes_tsgpu = boxes_ts.cuda()
            est_tsgpu = get_corner_box_3drot(boxes_tsgpu)
            self.assertTrue(np.allclose(est_tsgpu.cpu().numpy(), gt, atol=1e-2))
            torch.cuda.synchronize()
            t1 = time.time()
            for i in range(times):
                est_tsgpu = get_corner_box_3drot(boxes_tsgpu)
            torch.cuda.synchronize()
            t = (time.time()-t1) / times * 1000
            print(f"time is {t:.2f} ms {est_tsgpu.dtype}")

    def test_crop_pts_3drot(self):
        print("test_crop_pts_3drot")
        from det3.ops import crop_pts_3drot, read_npy
        from det3.dataloader.carladata import CarlaObj, CarlaObj, CarlaLabel, CarlaCalib
        pts = read_npy("./unit-test/data/test_CarlaAugmentor_000250.npy")
        calib = CarlaCalib("./unit-test/data/test_CarlaAugmentor_000250_calib.txt").read_calib_file()
        label = CarlaLabel("./unit-test/data/test_CarlaAugmentor_000250_label_imu.txt").read_label_file()
        pts = calib.lidar2imu(pts, key='Tr_imu_to_velo_top')
        for test_dtype in [np.float32, np.float64]:
            # obj.h, obj.w, obj.l, obj.x, obj.y, obj.z, obj.ry
            boxes = label.bboxes3d[:, [3,4,5,2,1,0,6]].astype(test_dtype)
            pts = pts.astype(test_dtype)
            gt = [obj.get_pts_idx(pts) for obj in label.data]
            gt = [np.where(itm)[0].flatten() for itm in gt]
            # np
            est = crop_pts_3drot(boxes, pts)
            self.assertTrue(len(gt) == len(est))
            for gt_, est_ in zip(gt, est):
                set_gt = set(gt_.tolist())
                set_est = set(est_.tolist())
                self.assertTrue(set_gt == set_est)
            times = 1000
            t1 = time.time()
            for i in range(times):
                est = crop_pts_3drot(boxes, pts)
            t = (time.time()-t1) / times * 1000
            print(f"np {times}: {t:.3f} ms {est[0].dtype}")
            # torch
            pts_ts = torch.from_numpy(pts)
            boxes_ts = torch.from_numpy(boxes)
            est_ts = crop_pts_3drot(boxes_ts, pts_ts)
            self.assertTrue(len(gt) == len(est_ts))
            for gt_, est_ts_ in zip(gt, est_ts):
                set_gt = set(gt_.tolist())
                set_est_ts = set(est_ts_.numpy().tolist())
                self.assertTrue(set_gt == set_est_ts)
            t1 = time.time()
            for i in range(times):
                est_ts = crop_pts_3drot(boxes_ts, pts_ts)
            t = (time.time()-t1) / times * 1000
            print(f"torchcpu {times}: {t:.3f} ms {est_ts[0].dtype}")
            # torchgpu
            pts_tsgpu = torch.from_numpy(pts).cuda()
            boxes_tsgpu = torch.from_numpy(boxes).cuda()
            est_tsgpu = crop_pts_3drot(boxes_tsgpu, pts_tsgpu)
            self.assertTrue(len(gt) == len(est_tsgpu))
            for gt_, est_tsgpu_ in zip(gt, est_tsgpu):
                set_gt = set(gt_.tolist())
                set_est_tsgpu = set(est_tsgpu_.cpu().numpy().tolist())
                self.assertTrue(set_gt == set_est_tsgpu)
            torch.cuda.synchronize()
            t1 = time.time()
            for i in range(times):
                est_tsgpu = crop_pts_3drot(boxes_tsgpu, pts_tsgpu)
            torch.cuda.synchronize()
            t = (time.time()-t1) / times * 1000
            print(f"torchgpu {times}: {t:.3f} ms {est_tsgpu[0].dtype}")

if __name__ == "__main__":
    unittest.main()