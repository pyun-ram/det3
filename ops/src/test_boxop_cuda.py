import torch
import numpy as np
import boxop_cuda
from det3.dataloader.carladata import CarlaObj

if __name__ == "__main__":
    x = np.linspace(-2, 2, 5)
    y = np.linspace(-3, 3, 7)
    z = np.linspace(-4, 4, 9)
    xv, yv, zv = np.meshgrid(x, y, z)
    pts = np.hstack([xv.flatten().reshape(-1, 1),
                     yv.flatten().reshape(-1, 1),
                     zv.flatten().reshape(-1, 1)]).astype(np.float32)
    pts = [pts for i in range(10)]
    pts = np.vstack(pts)
    boxes = np.array([[0, 0, 0, 3, 4, 5, 0.3],
                      [0, 0, 0, 3, 4, 5, 0]]).astype(np.float32)
    gt = []
    for box in boxes:
        obj = CarlaObj()
        obj.x, obj.y, obj.z, obj.l, obj.w, obj.h, obj.ry = box.flatten()
        idx = obj.get_pts_idx(pts)
        gt.append(idx)
    gt = np.vstack(gt)
    
    boxes_tsgpu = torch.from_numpy(boxes).cuda()
    pts_tsgpu = torch.from_numpy(pts).cuda()
    res = boxop_cuda.crop_pts_3drot(boxes_tsgpu, pts_tsgpu)
    print(res.shape, gt.shape)
    print(np.array_equal(res.cpu().numpy(), gt))
    
