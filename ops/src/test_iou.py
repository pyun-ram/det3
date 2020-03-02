import torch
import iou_cpp
from det3.utils.utils import compute_intersec

if __name__ == "__main__":
    import numpy as np
    box = np.array([3, 2, 4, 4.5])
    others = np.array([[6, 3, 2, 5], [5, 3, 2, 5]])
    others = [others + np.abs(np.random.randn(8).reshape(2, 4)) for i in range(2)]
    others = np.vstack(others)
    box_ts = torch.from_numpy(box).float()
    others_ts = torch.from_numpy(others).float()
    est_ts = iou_cpp.compute_intersect_2d_cpu(box_ts, others_ts)
    box_gt = np.array([box[0], box[1], 0, box[2], box[3], 0, 0])
    others_gt = [np.array([itm[0], itm[1], 0, itm[2], itm[3], 0, 0]) for itm in others]
    others_gt = np.vstack(others_gt)
    gt = compute_intersec(box_gt, others_gt, mode="2d-rot")
    print(est_ts, gt)