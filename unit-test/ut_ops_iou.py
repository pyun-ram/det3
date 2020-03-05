'''
 File Created: Mon Mar 02 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
'''
import torch
import unittest
import numpy as np

class UTIOU(unittest.TestCase):
    def test_compute_intersect_2d(self):
        import time
        from det3.ops import compute_intersect_2d, write_pkl
        from det3.utils.utils import compute_intersec
        for test_dtype in [np.float32, np.float64]:
            # npy
            boxes = np.array([[3, 2, 4, 4.5]])
            boxes = [boxes + np.abs(np.random.randn(8).reshape(2, 4)) for i in range(10)]
            boxes = np.vstack(boxes).astype(test_dtype)
            others = np.array([[6, 3, 2, 5]])
            others = [others + np.abs(np.random.randn(8).reshape(2, 4)) for i in range(21)]
            others = np.vstack(others).astype(test_dtype)
            boxes_gt = [np.array([itm[0], itm[1], 0, itm[2], itm[3], 0, 0]) for itm in boxes]
            boxes_gt = np.vstack(boxes_gt)
            others_gt = [np.array([itm[0], itm[1], 0, itm[2], itm[3], 0, 0]) for itm in others]
            others_gt = np.vstack(others_gt)
            gt = []
            for box_gt in boxes_gt:
                gt.append(compute_intersec(box_gt, others_gt, mode="2d-rot"))
            gt = np.stack(gt, axis=0)
            times = 100

            est = compute_intersect_2d(boxes, others)
            self.assertTrue(est.shape == gt.shape)
            self.assertTrue(np.allclose(est, gt, atol=1e-1))
            self.assertTrue(est.dtype == boxes.dtype)
            t1 = time.time()
            for i in range(times):
                est = compute_intersect_2d(boxes, others)
            t = (time.time()-t1) / times * 1000
            print(f"compute_intersect_2d np: {t:.2f} ms")
            # torch
            boxes_ts = torch.from_numpy(boxes)
            others_ts = torch.from_numpy(others)
            est_ts = compute_intersect_2d(boxes_ts, others_ts)
            self.assertTrue(est_ts.numpy().shape == gt.shape)
            self.assertTrue(np.allclose(est_ts.numpy(), gt, atol=1e-1))
            self.assertTrue(est_ts.dtype == boxes_ts.dtype)
            t1 = time.time()
            for i in range(times):
                est_ts = compute_intersect_2d(boxes_ts, others_ts)
            t = (time.time()-t1) / times * 1000
            print(f"compute_intersect_2d torch: {t:.2f} ms")
            # torch cuda
            boxes_tsgpu = boxes_ts.cuda()
            others_tsgpu = others_ts.cuda()
            est_tsgpu = compute_intersect_2d(boxes_tsgpu, others_tsgpu)
            self.assertTrue(est_tsgpu.cpu().numpy().shape == gt.shape)
            self.assertTrue(np.allclose(est_tsgpu.cpu().numpy(), gt, atol=1e-1))
            self.assertTrue(est_tsgpu.dtype == boxes_tsgpu.dtype)
            torch.cuda.synchronize()
            t1 = time.time()
            for i in range(times):
                est_tsgpu = compute_intersect_2d(boxes_tsgpu, others_tsgpu)
            torch.cuda.synchronize()
            t = (time.time()-t1) / times * 1000
            print(f"compute_intersect_2d torch cuda: {t:.2f} ms")

    def test_compute_intersect_2drot(self):
        import time
        from det3.ops import compute_intersect_2drot, write_pkl
        from det3.utils.utils import compute_intersec
        for test_dtype in [np.float32, np.float64]:
            # npy
            boxes = np.array([[3, 2, 4, 4.5, 0.1]])
            boxes = [boxes + np.abs(np.random.randn(10).reshape(2, 5)) for i in range(10)]
            boxes = np.vstack(boxes).astype(test_dtype)
            others = np.array([[6, 3, 2, 5, 0.05]])
            others = [others + np.abs(np.random.randn(10).reshape(2, 5)) for i in range(21)]
            others = np.vstack(others).astype(test_dtype)
            boxes_gt = [np.array([itm[0], itm[1], 0, itm[2], itm[3], 0, itm[4]]) for itm in boxes]
            boxes_gt = np.vstack(boxes_gt)
            others_gt = [np.array([itm[0], itm[1], 0, itm[2], itm[3], 0, itm[4]]) for itm in others]
            others_gt = np.vstack(others_gt)
            gt = []
            for box_gt in boxes_gt:
                gt.append(compute_intersec(box_gt, others_gt, mode="2d-rot"))
            gt = np.stack(gt, axis=0)
            times = 100
            est = compute_intersect_2drot(boxes, others)
            self.assertTrue(est.shape == gt.shape)
            self.assertTrue(np.allclose(est, gt, atol=1e-1))
            self.assertTrue(est.dtype == boxes.dtype)
            t1 = time.time()
            for i in range(times):
                est = compute_intersect_2drot(boxes, others)
            t = (time.time()-t1) / times * 1000
            print(f"compute_intersect_2drot np: {t:.2f} ms")
            # torch
            boxes_ts = torch.from_numpy(boxes)
            others_ts = torch.from_numpy(others)
            est_ts = compute_intersect_2drot(boxes_ts, others_ts)
            self.assertTrue(est_ts.numpy().shape == gt.shape)
            self.assertTrue(np.allclose(est_ts.numpy(), gt, atol=1e-1))
            self.assertTrue(est_ts.dtype == boxes_ts.dtype)
            t1 = time.time()
            for i in range(times):
                est_ts = compute_intersect_2drot(boxes_ts, others_ts)
            t = (time.time()-t1) / times * 1000
            print(f"compute_intersect_2drot torch: {t:.2f} ms")
            # torch cuda
            boxes_tsgpu = torch.from_numpy(boxes).cuda()
            others_tsgpu = torch.from_numpy(others).cuda()
            est_tsgpu = compute_intersect_2drot(boxes_tsgpu, others_tsgpu)
            self.assertTrue(est_tsgpu.cpu().numpy().shape == gt.shape)
            self.assertTrue(np.allclose(est_tsgpu.cpu().numpy(), gt, atol=1e-1))
            self.assertTrue(est_tsgpu.dtype == boxes_tsgpu.dtype)
            torch.cuda.synchronize()
            t1 = time.time()
            for i in range(times):
                est_tsgpu = compute_intersect_2drot(boxes_tsgpu, others_tsgpu)
            torch.cuda.synchronize()
            t = (time.time()-t1) / times * 1000
            print(f"compute_intersect_2drot torch cuda: {t:.2f} ms")
        


if __name__ == "__main__":
    unittest.main()