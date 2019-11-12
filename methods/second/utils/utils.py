import numpy as np
from det3.dataloader.kittidata import KittiLabel
def filt_label_by_cls(label, keep_cls):
    label.data = list(filter(lambda obj: obj.type in keep_cls, label.data))
    res = KittiLabel()
    for obj in label.data:
        res.add_obj(obj)
    res.current_frame = label.current_frame
    return res

def is_array_like(x):
    "Source: https://github.com/traveller59/second.pytorch"
    return isinstance(x, (list, tuple, np.ndarray))

def shape_mergeable(x, expected_shape):
    "Source: https://github.com/traveller59/second.pytorch"
    mergeable = True
    if is_array_like(x) and is_array_like(expected_shape):
        x = np.array(x)
        if len(x.shape) == len(expected_shape):
            for s, s_ex in zip(x.shape, expected_shape):
                if s_ex is not None and s != s_ex:
                    mergeable = False
                    break
    return mergeable