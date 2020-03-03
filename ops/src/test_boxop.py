import torch
import boxop_cpp

if __name__ == "__main__":
    import numpy as np
    from det3.dataloader.carladata import CarlaObj
    boxes = np.array([[6, 3, 2, 5, np.pi * 0.1], [3, 2, 4, 4.5, np.pi * 0.2]])
    boxes = [boxes + np.abs(np.random.randn(10).reshape(2, 5)) for i in range(15)]
    boxes = np.vstack(boxes)
    boxes_ts = torch.from_numpy(boxes).float().cuda()
    est = boxop_cpp.get_corner_box_2drot(boxes_ts)
    gt = []
    for box in boxes:
        obj = CarlaObj()
        obj.x, obj.y, obj.l, obj.w, obj.ry = box.flatten()
        obj.z, obj.h =0, 0
        gt.append(obj.get_bbox3dcorners()[:4, :2])
    gt = np.stack(gt, axis=0).astype(np.float32)
    print(np.allclose(est.cpu().numpy(), gt, atol=1e-2))