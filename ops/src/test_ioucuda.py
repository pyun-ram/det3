import torch
import iou_cuda
from det3.utils.utils import compute_intersec
from det3.ops import compute_intersect_2drot
def compute_intersect_2drot_torchgpu(boxes, others):
    return iou_cuda.compute_intersect_2drot(others, boxes).T

if __name__ == "__main__":
    import numpy as np
    box = np.array([[4, 2, 4, 4.5, 0.1], [3, 3, 4, 4.5, 0.3]])
    box = [box + np.abs(np.random.randn(10).reshape(2, 5)) for i in range(50)]
    box = np.vstack(box)
    others = np.array([[6, 3, 2, 5, 0.2], [5, 3, 2, 5, 0.4]])
    others = [others + np.abs(np.random.randn(10).reshape(2, 5)) for i in range(50)]
    others = np.vstack(others)
    box_tsgpu = torch.from_numpy(box).float().cuda()
    others_tsgpu = torch.from_numpy(others).float().cuda()
    est_tsgpu = compute_intersect_2drot_torchgpu(box_tsgpu, others_tsgpu)
    gt = compute_intersect_2drot(box_tsgpu.cpu().numpy(), others_tsgpu.cpu().numpy())
    # box_gt = np.array([box[0], box[1], 0, box[2], box[3], 0, 0])
    # others_gt = [np.array([itm[0], itm[1], 0, itm[2], itm[3], 0, 0]) for itm in others]
    # others_gt = np.vstack(others_gt)
    # gt = compute_intersec(box_gt, others_gt, mode="2d-rot")
    print(est_tsgpu.shape)
    print(gt.shape)
    print(np.allclose(est_tsgpu.cpu().numpy(), gt))