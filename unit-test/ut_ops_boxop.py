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
        from det3.ops import get_corner_box_2drot
        from det3.dataloader.carladata import CarlaObj
        # np
        boxes = np.array([[6, 3, 2, 5, np.pi * 0.1], [3, 2, 4, 4.5, np.pi * 0.2]])
        boxes = [boxes + np.abs(np.random.randn(10).reshape(2, 5)) for i in range(15)]
        boxes = np.vstack(boxes)
        est = get_corner_box_2drot(boxes)
        gt = []
        for box in boxes:
            obj = CarlaObj()
            obj.x, obj.y, obj.l, obj.w, obj.ry = box.flatten()
            obj.z, obj.h =0, 0
            gt.append(obj.get_bbox3dcorners()[:4, :2])
        gt = np.stack(gt, axis=0)
        self.assertTrue(np.array_equal(est, gt))
        # times = 100
        # t1 = time.time()
        # for i in range(times):
        #     est = get_corner_box_2drot(boxes)
        # print(f"time is {time.time()-t1:.3f} s for {times} times")
        # torch
        boxes_ts = torch.from_numpy(boxes).float()
        est_ts = get_corner_box_2drot(boxes_ts)
        self.assertTrue(np.allclose(est_ts.numpy(), gt, atol=1e-2))
        # times = 100
        # t1 = time.time()
        # for i in range(times):
        #     est_ts = get_corner_box_2drot(boxes_ts)
        # print(f"time is {time.time()-t1:.3f} s for {times} times")
        # torch gpu
        boxes_tsgpu = boxes_ts.cuda().double()
        est_tsgpu = get_corner_box_2drot(boxes_tsgpu)
        self.assertTrue(np.allclose(est_tsgpu.cpu().numpy(), gt, atol=1e-2))
        # times = 100
        # torch.cuda.synchronize()
        # t1 = time.time()
        # for i in range(times):
        #     est_tsgpu = get_corner_box_2drot(boxes_tsgpu)
        # torch.cuda.synchronize()
        # print(f"time is {time.time()-t1:.3f} s for {times} times")

    def test_get_corner_box_3drot(self):
        from det3.ops import get_corner_box_3drot
        from det3.dataloader.carladata import CarlaObj
        # np
        boxes = np.array([[6, 3, 0, 2, 5, 1, np.pi * 0.1], [3, 2, 1, 4, 4.5, 2, np.pi * 0.2]])
        boxes = [boxes + np.abs(np.random.randn(14).reshape(2, 7)) for i in range(15)]
        boxes = np.vstack(boxes)
        est = get_corner_box_3drot(boxes)
        gt = []
        for box in boxes:
            obj = CarlaObj()
            obj.x, obj.y, obj.z, obj.l, obj.w, obj.h, obj.ry = box.flatten()
            gt.append(obj.get_bbox3dcorners())
        gt = np.stack(gt, axis=0)
        self.assertTrue(np.array_equal(est, gt))
        # times = 100
        # t1 = time.time()
        # for i in range(times):
        #     est = get_corner_box_3drot(boxes)
        # print(f"time is {time.time()-t1:.3f} s for {times} times")
        # torch
        boxes_ts = torch.from_numpy(boxes).float()
        est_ts = get_corner_box_3drot(boxes_ts)
        self.assertTrue(np.allclose(est_ts.numpy(), gt, atol=1e-2))
        # times = 100
        # t1 = time.time()
        # for i in range(times):
        #     est_ts = get_corner_box_3drot(boxes_ts)
        # print(f"time is {time.time()-t1:.3f} s for {times} times")
        # torch gpu
        boxes_tsgpu = boxes_ts.cuda().float()
        est_tsgpu = get_corner_box_3drot(boxes_tsgpu)
        self.assertTrue(np.allclose(est_tsgpu.cpu().numpy(), gt, atol=1e-2))
        # times = 100
        # torch.cuda.synchronize()
        # t1 = time.time()
        # for i in range(times):
        #     est_tsgpu = get_corner_box_3drot(boxes_tsgpu)
        # torch.cuda.synchronize()
        # print(f"time is {time.time()-t1:.3f} s for {times} times")

if __name__ == "__main__":
    unittest.main()