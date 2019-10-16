import numpy as np
class BaseAnchorGenerator:
    @property
    def num_anchors_per_localization(self):
        raise NotImplementedError

    def generate(self, feature_map_size):
        raise NotImplementedError

    @property 
    def ndim(self):
        raise NotImplementedError

class AnchorGeneratorBEV(BaseAnchorGenerator):
    '''
    Source: https://github.com/traveller59/second.pytorch
    '''
    def __init__(self,
                 anchor_ranges,
                 match_threshold,
                 unmatch_threshold,
                 class_name,
                 sizes=[1.6, 3.9, 1.56],
                 rotations=[0, np.pi / 2],
                 custom_values=(),
                 dtype=np.float32):
        self._sizes = sizes
        self._anchor_ranges = anchor_ranges
        self._rotations = rotations
        self._dtype = dtype
        self._custom_values = custom_values
        self.match_threshold = match_threshold
        self.unmatch_threshold = unmatch_threshold
        self._class_name = class_name

    @property
    def class_name(self):
        return self._class_name

    def generate(self, feature_map_size):
        from det3.methods.second.ops.ops import create_anchors_3d_range
        res = create_anchors_3d_range(
            feature_map_size, self._anchor_ranges, self._sizes,
            self._rotations, self._dtype)

        if len(self._custom_values) > 0:
            custom_ndim = len(self._custom_values)
            custom = np.zeros([*res.shape[:-1], custom_ndim])
            custom[:] = self._custom_values
            res = np.concatenate([res, custom], axis=-1)
        return res

    @property
    def num_anchors_per_localization(self):
        num_rot = len(self._rotations)
        num_size = np.array(self._sizes).reshape([-1, 3]).shape[0]
        return num_rot * num_size

    @property 
    def ndim(self):
        return 7 + len(self._custom_values)

    @property 
    def custom_ndim(self):
        return len(self._custom_values)

if __name__=="__main__":
    anchor_ranges= [0, -40.0, -1.00, 70.4, 40.0, -1.00]
    sizes=[1.6, 3.9, 1.56]
    rotations=[0, 1.57]
    anchor_generator_est = AnchorGeneratorBEV(anchor_ranges, sizes, rotations)
    est = anchor_generator_est.generate(feature_map_size=[1, 200, 176])
    # gt = np.load("./gt_anchor_generator.npy")
    # print((est == gt).all())
    print(est.shape)