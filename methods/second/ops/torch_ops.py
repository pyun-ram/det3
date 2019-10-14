import torch
import numpy as np
from det3.methods.second.ops.ops import (center_to_corner_box2d,
                                         corner_to_standup_nd,
                                         iou_jit)
from spconv.utils import rotate_non_max_suppression_cpu

def rotate_nms(rbboxes,
               scores,
               pre_max_size=None,
               post_max_size=None,
               iou_threshold=0.5):
    if pre_max_size is not None:
        num_keeped_scores = scores.shape[0]
        pre_max_size = min(num_keeped_scores, pre_max_size)
        scores, indices = torch.topk(scores, k=pre_max_size)
        rbboxes = rbboxes[indices]
    dets = torch.cat([rbboxes, scores.unsqueeze(-1)], dim=1)
    dets_np = dets.data.cpu().numpy()
    if len(dets_np) == 0:
        keep = np.array([], dtype=np.int64)
    else:
        ret = np.array(rotate_nms_cc(dets_np, iou_threshold), dtype=np.int64)
        keep = ret[:post_max_size]
    if keep.shape[0] == 0:
        return torch.zeros([0]).long().to(rbboxes.device)
    if pre_max_size is not None:
        keep = torch.from_numpy(keep).long().to(rbboxes.device)
        return indices[keep]
    else:
        return torch.from_numpy(keep).long().to(rbboxes.device)

def rotate_nms_cc(dets, thresh):
    scores = dets[:, 5]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    dets_corners = center_to_corner_box2d(dets[:, :2], dets[:, 2:4],
                                                     dets[:, 4])

    dets_standup = corner_to_standup_nd(dets_corners)

    standup_iou = iou_jit(dets_standup, dets_standup, eps=0.0)
    # print(dets_corners.shape, order.shape, standup_iou.shape)
    return rotate_non_max_suppression_cpu(dets_corners, order, standup_iou,
                                          thresh)