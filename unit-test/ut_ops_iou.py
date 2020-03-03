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
        # npy
        boxes = np.array([[3, 2, 4, 4.5], [3, 2, 3, 4.5]])
        others = np.array([[6, 3, 2, 5], [5, 3, 2, 5]])
        others = [others + np.abs(np.random.randn(8).reshape(2, 4)) for i in range(50)]
        others = np.vstack(others)
        est = compute_intersect_2d(boxes, others)
        boxes_gt = [np.array([itm[0], itm[1], 0, itm[2], itm[3], 0, 0]) for itm in boxes]
        boxes_gt = np.vstack(boxes_gt)
        others_gt = [np.array([itm[0], itm[1], 0, itm[2], itm[3], 0, 0]) for itm in others]
        others_gt = np.vstack(others_gt)
        gt = []
        for box_gt in boxes_gt:
            gt.append(compute_intersec(box_gt, others_gt, mode="2d-rot"))
        gt = np.stack(gt, axis=0)
        self.assertTrue(est.shape == gt.shape)
        self.assertTrue(np.allclose(est, gt, atol=1e-1))
        # torch
        boxes_ts = torch.from_numpy(boxes)
        others_ts = torch.from_numpy(others)
        est_ts = compute_intersect_2d(boxes_ts, others_ts)
        self.assertTrue(est_ts.numpy().shape == gt.shape)
        self.assertTrue(np.allclose(est_ts.numpy(), gt, atol=1e-1))
        # torch cuda
        boxes_tsgpu = boxes_ts.cuda()
        others_tsgpu = others_ts.cuda()
        est_tsgpu = compute_intersect_2d(boxes_tsgpu, others_tsgpu)
        self.assertTrue(est_tsgpu.cpu().numpy().shape == gt.shape)
        self.assertTrue(np.allclose(est_tsgpu.cpu().numpy(), gt, atol=1e-1))

    def test_compute_intersect_2drot(self):
        import time
        from det3.ops import compute_intersect_2drot, write_pkl
        from det3.utils.utils import compute_intersec
        # npy
        boxes = np.array([[3, 2, 4, 4.5, 0.1], [3, 2, 3, 4.5, 0.5]])
        others = np.array([[6, 3, 2, 5, 0.05], [5, 3, 2, 5, 0.2]])
        others = [others + np.abs(np.random.randn(10).reshape(2, 5)) for i in range(50)]
        others = np.vstack(others)
        est = compute_intersect_2drot(boxes, others)
        boxes_gt = [np.array([itm[0], itm[1], 0, itm[2], itm[3], 0, itm[4]]) for itm in boxes]
        boxes_gt = np.vstack(boxes_gt)
        others_gt = [np.array([itm[0], itm[1], 0, itm[2], itm[3], 0, itm[4]]) for itm in others]
        others_gt = np.vstack(others_gt)
        gt = []
        for box_gt in boxes_gt:
            gt.append(compute_intersec(box_gt, others_gt, mode="2d-rot"))
        gt = np.stack(gt, axis=0)
        self.assertTrue(est.shape == gt.shape)
        self.assertTrue(np.allclose(est, gt, atol=1e-1))
        # torch
        boxes_ts = torch.from_numpy(boxes)
        others_ts = torch.from_numpy(others)
        est_ts = compute_intersect_2drot(boxes_ts, others_ts)
        boxes_gt = [np.array([itm[0], itm[1], 0, itm[2], itm[3], 0, itm[4]]) for itm in boxes]
        boxes_gt = np.vstack(boxes_gt)
        others_gt = [np.array([itm[0], itm[1], 0, itm[2], itm[3], 0, itm[4]]) for itm in others]
        others_gt = np.vstack(others_gt)
        gt = []
        for box_gt in boxes_gt:
            gt.append(compute_intersec(box_gt, others_gt, mode="2d-rot"))
        gt = np.stack(gt, axis=0)
        self.assertTrue(est_ts.numpy().shape == gt.shape)
        self.assertTrue(np.allclose(est_ts.numpy(), gt, atol=1e-1))


if __name__ == "__main__":
    unittest.main()