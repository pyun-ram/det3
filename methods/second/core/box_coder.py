import numpy as np
from det3.dataloader.kittidata import KittiLabel
from det3.methods.second.utils.log_tool import Logger

class BaseBoxCoder:
    def __init__(self):
        raise NotImplementedError
    def encode(label):
        raise NotImplementedError
    def decode(grid):
        raise NotImplementedError
class BoxCoderV1(BaseBoxCoder):
    "Source: https://github.com/traveller59/second.pytorch"
    def __init__(self, custom_ndim=0):
        self.custom_ndim = custom_ndim
        Logger.log_txt("Warning: BoxCoderV1 requires unit-test.")

    def encode(self, boxes:np.ndarray, anchors:np.ndarray) -> np.ndarray:
        """box encode for VoxelNet in lidar
        Args:
            boxes ([N, 7 + ?] Tensor): normal boxes: x, y, z, w, l, h, r, custom values
            anchors ([N, 7] Tensor): anchors
        """
        from det3.methods.second.ops.ops import second_box_encode
        return second_box_encode(boxes, anchors)

    def decode(self, encodings, anchors):
        from det3.methods.second.ops.ops import second_box_decode
        return second_box_decode(encodings, anchors)

    @property
    def code_size(self):
        return self.custom_ndim + 7
