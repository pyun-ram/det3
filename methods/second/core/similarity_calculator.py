from det3.methods.second.utils.log_tool import Logger
class BaseSimilarityCalculator:
    def __init__(self):
        raise NotImplementedError

    def compare(self, boxes1, boxes2):
        raise NotImplementedError

class NearestIoUSimilarity(BaseSimilarityCalculator):
    def __init__(self):
        Logger.log_txt("Warning: NearestIoUSimilarity requires unit-test.")
        Logger.log_txt("Warning: NearestIoUSimilarity can be improved by shaply implementation.")
        pass

    def compare(self, boxes1, boxes2):
        """Compute matrix of (negated) sq distances.

        Args:
        boxlist1: BoxList holding N boxes.
        boxlist2: BoxList holding M boxes.

        Returns:
        A tensor with shape [N, M] representing negated pairwise squared distance.
        """
        from det3.methods.second.ops.ops import rbbox2d_to_near_bbox, iou_jit
        boxes1_bv = rbbox2d_to_near_bbox(boxes1)
        boxes2_bv = rbbox2d_to_near_bbox(boxes2)
        ret = iou_jit(boxes1_bv, boxes2_bv, eps=0.0)
        return ret