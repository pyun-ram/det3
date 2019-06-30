'''
File Created: Sunday, 24th March 2019 4:59:19 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import unittest
import numpy as np
try:
    from ..utils.utils import apply_R, apply_tr, roty
except:
    # Run script python3 dataloader/data.py
    import sys
    sys.path.append("../")
    from det3.utils.utils import apply_R, apply_tr, roty

class TestUtils(unittest.TestCase):
    def test_applyR(self):
        pts = np.array([[-0.6, 0.,  0.24],
                        [ 0.6, 0.,  0.24],
                        [ 0.6, 0., -0.24],
                        [-0.6, 0., -0.24]])
        ry = 0.01
        ans = np.array([[-0.59757004, 0,  0.2459879 ],
                        [ 0.60236996, 0,  0.2339881 ],
                        [ 0.59757004, 0, -0.2459879 ],
                        [-0.60236996, 0, -0.2339881 ]])
        self.assertTrue(np.allclose(apply_R(pts, roty(ry)), ans, rtol=1e-5))
    def test_applyT(self):
        pts = np.array([[-0.6, 0.,  0.24],
                        [ 0.6, 0.,  0.24],
                        [ 0.6, 0., -0.24],
                        [-0.6, 0., -0.24]])
        t = np.array([1,2,3]).reshape(1,3)
        ans = pts + t
        self.assertTrue(np.allclose(apply_tr(pts, t), ans, rtol=1e-5))
# TODO: TEST clip_ry
# TODO: TEST NMS & its dependent functions
if __name__ == "__main__":
    unittest.main()