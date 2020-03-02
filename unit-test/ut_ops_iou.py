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
        box = np.array([3, 2, 4, 4.5])
        others = np.array([[6, 3, 2, 5], [5, 3, 2, 5]])
        others = [others + np.abs(np.random.randn(8).reshape(2, 4)) for i in range(50)]
        others = np.vstack(others)
        est = compute_intersect_2d(box, others)
        box_gt = np.array([box[0], box[1], 0, box[2], box[3], 0, 0])
        others_gt = [np.array([itm[0], itm[1], 0, itm[2], itm[3], 0, 0]) for itm in others]
        others_gt = np.vstack(others_gt)
        gt = compute_intersec(box_gt, others_gt, mode="2d-rot")
        self.assertTrue(np.allclose(est, gt, atol=1e-1))
        # times = 1000
        # t1 = time.time()
        # for i in range(times):
        #     est = compute_intersect_2d(box, others)
        # print(f"time is {time.time()-t1:.3f} s for {times} times")
        # torch
        box_ts = torch.from_numpy(box)
        others_ts = torch.from_numpy(others)
        est_ts = compute_intersect_2d(box_ts, others_ts)
        self.assertTrue(np.allclose(est_ts.numpy(), gt, atol=1e-1))
        # times = 1000
        # t1 = time.time()
        # for i in range(times):
        #     est_ts = compute_intersect_2d(box_ts, others_ts)
        # print(f"time is {time.time()-t1:.3f} s for {times} times")
        # torch cuda
        box_tsgpu = torch.from_numpy(box).cuda()
        others_tsgpu = torch.from_numpy(others).cuda()
        est_tsgpu = compute_intersect_2d(box_tsgpu, others_tsgpu)
        self.assertTrue(np.allclose(est_tsgpu.cpu().numpy(), gt, atol=1e-1))
        # times = 1000
        # torch.cuda.synchronize()
        # t1 = time.time()
        # for i in range(times):
        #     est_tsgpu = compute_intersect_2d(box_tsgpu, others_tsgpu)
        # torch.cuda.synchronize()
        # print(f"time is {time.time()-t1:.3f} s for {times} times")
        

if __name__ == "__main__":
    unittest.main()