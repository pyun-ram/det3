'''
File Created: Tuesday, 19th March 2019 9:04:16 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import unittest
import numpy as np
try:
    from ..dataloarder.data import KittiCalib, KittiObj, KittiLabel
except:
    # Run script python3 dataloader/data.py
    import sys
    sys.path.append("../")
    from det3.dataloarder.data import KittiCalib, KittiObj, KittiLabel

class TestKittiCalib(unittest.TestCase):
    def test_init(self):
        kitticalib = KittiCalib("./unit-test/data/test_KittiCalib_000000.txt")
        self.assertEqual(kitticalib.data, None)
        self.assertEqual(kitticalib.R0_rect, None)
        self.assertEqual(kitticalib.Tr_velo_to_cam, None)
    def test_read_kitti_calib_file(self):
        calib = KittiCalib("./unit-test/data/test_KittiCalib_000000.txt").read_kitti_calib_file()
        self.assertTrue('P0' in calib.data.keys())
        self.assertTrue('P1' in calib.data.keys())
        self.assertTrue('P2' in calib.data.keys())
        self.assertTrue('P3' in calib.data.keys())
        self.assertTrue('R0_rect' in calib.data.keys())
        self.assertTrue('Tr_velo_to_cam' in calib.data.keys())
        self.assertTrue('Tr_imu_to_velo' in calib.data.keys())
        for k, v in calib.data.items():
            for itm in v:
                self.assertTrue(isinstance(itm, float))
        self.assertEqual(calib.R0_rect.shape, (4, 4))
        self.assertEqual(calib.Tr_velo_to_cam.shape, (4, 4))
    def test_leftcam2lidar(self):
        calib = KittiCalib("./unit-test/data/test_KittiCalib_000000.txt").read_kitti_calib_file()
        pts = np.array([[1.24242996, 1.47, 8.6559879],
                        [2.44236996, 1.47, 8.6439881],
                        [2.43757004, 1.47, 8.1640121],
                        [1.23763004, 1.47, 8.1760119]])
        ans = np.array([[8.97831988, -1.25877336, -1.59332772],
                        [8.96440504, -2.45859462, -1.60867200],
                        [8.48444395, -2.45306157, -1.60607095],
                        [8.49835880, -1.25324031, -1.59072667]])
        self.assertTrue(np.allclose(calib.leftcam2lidar(pts), ans, rtol=1e-5))
    def test_lidar2leftcam(self):
        calib = KittiCalib("./unit-test/data/test_KittiCalib_000000.txt").read_kitti_calib_file()
        pts = np.array([[1.24242996, 1.47, 8.6559879],
                        [2.44236996, 1.47, 8.6439881],
                        [2.43757004, 1.47, 8.1640121],
                        [1.23763004, 1.47, 8.1760119]])
        self.assertTrue(np.allclose(calib.lidar2leftcam(calib.leftcam2lidar(pts)), pts, rtol=1e-5))

class TestKittiObj(unittest.TestCase):
    def test_init(self):
        kittiobj = KittiObj('Car 0.00 0 1.55 614.24 181.78 727.31 284.77 1.57 1.73 4.15 1.00 1.75 13.22 1.62')
        self.assertEqual(kittiobj.type, 'Car')
        self.assertEqual(kittiobj.truncated, 0)
        self.assertEqual(kittiobj.occluded, 0)
        self.assertEqual(kittiobj.alpha, 1.55)
        self.assertEqual(kittiobj.bbox_l, 614.24)
        self.assertEqual(kittiobj.bbox_t, 181.78)
        self.assertEqual(kittiobj.bbox_r, 727.31)
        self.assertEqual(kittiobj.bbox_b, 284.77)
        self.assertEqual(kittiobj.h, 1.57)
        self.assertEqual(kittiobj.w, 1.73)
        self.assertEqual(kittiobj.l, 4.15)
        self.assertEqual(kittiobj.x, 1.00)
        self.assertEqual(kittiobj.y, 1.75)
        self.assertEqual(kittiobj.z, 13.22)
        self.assertEqual(kittiobj.ry, 1.62)
        self.assertEqual(kittiobj.score, None)
        kittiobj = KittiObj('Car 0.00 0 1.55 614.24 181.78 727.31 284.77 1.57 1.73 4.15 1.00 1.75 13.22 1.62 0.99')
        self.assertEqual(kittiobj.type, 'Car')
        self.assertEqual(kittiobj.truncated, 0)
        self.assertEqual(kittiobj.occluded, 0)
        self.assertEqual(kittiobj.alpha, 1.55)
        self.assertEqual(kittiobj.bbox_l, 614.24)
        self.assertEqual(kittiobj.bbox_t, 181.78)
        self.assertEqual(kittiobj.bbox_r, 727.31)
        self.assertEqual(kittiobj.bbox_b, 284.77)
        self.assertEqual(kittiobj.h, 1.57)
        self.assertEqual(kittiobj.w, 1.73)
        self.assertEqual(kittiobj.l, 4.15)
        self.assertEqual(kittiobj.x, 1.00)
        self.assertEqual(kittiobj.y, 1.75)
        self.assertEqual(kittiobj.z, 13.22)
        self.assertEqual(kittiobj.ry, 1.62)
        self.assertEqual(kittiobj.score, 0.99)
    def test_get3dcorners(self):
        kittiobj = KittiObj('Pedestrian 0.0 0.0 -0.2 712.4 143.0 810.73 307.92 1.89 0.48 1.2 1.84 1.47 8.41 0.01')
        ans = np.array([[1.24242996, 1.47, 8.6559879],
                        [2.44236996, 1.47, 8.6439881],
                        [2.43757004, 1.47, 8.1640121],
                        [1.23763004, 1.47, 8.1760119],
                        [1.24242996, -0.42, 8.6559879],
                        [2.44236996, -0.42, 8.6439881],
                        [2.43757004, -0.42, 8.1640121],
                        [1.23763004, -0.42, 8.1760119]])
        self.assertTrue(np.allclose(kittiobj.get_bbox3dcorners(), ans, rtol=1e-5))
    # TODO:TEST from_corners
    def test_from_corners(self):
        kittiobj = KittiObj()
        calib = KittiCalib("./unit-test/data/test_KittiCalib_000000.txt").read_kitti_calib_file()
        cns = np.array([[1.24242996, 1.47, 8.6559879],
                [2.44236996, 1.47, 8.6439881],
                [2.43757004, 1.47, 8.1640121],
                [1.23763004, 1.47, 8.1760119],
                [1.24242996, -0.42, 8.6559879],
                [2.44236996, -0.42, 8.6439881],
                [2.43757004, -0.42, 8.1640121],
                [1.23763004, -0.42, 8.1760119]])
        kittiobj.from_corners(calib, cns, 'Pedestrian', 1.0)
        self.assertEqual(kittiobj.type, 'Pedestrian')
        self.assertEqual(kittiobj.truncated, 0)
        self.assertEqual(kittiobj.occluded, 0)
        self.assertEqual(kittiobj.alpha, 0)
        self.assertTrue(np.allclose(kittiobj.bbox_l, 712.4,  rtol=0.1))
        self.assertTrue(np.allclose(kittiobj.bbox_t, 143.0,  rtol=0.1))
        self.assertTrue(np.allclose(kittiobj.bbox_r, 810.73, rtol=0.1))
        self.assertTrue(np.allclose(kittiobj.bbox_b, 307.92, rtol=0.1))        
        self.assertTrue(np.allclose(kittiobj.h, 1.89, rtol=1e-5))
        self.assertTrue(np.allclose(kittiobj.w, 0.48, rtol=1e-5))
        self.assertTrue(np.allclose(kittiobj.l, 1.2, rtol=1e-5))
        self.assertTrue(np.allclose(kittiobj.x, 1.84, rtol=1e-5))
        self.assertTrue(np.allclose(kittiobj.y, 1.47, rtol=1e-5))
        self.assertTrue(np.allclose(kittiobj.z, 8.41, rtol=1e-5))
        self.assertTrue(np.allclose(kittiobj.ry, 0.01, rtol=1e-5))
        self.assertEqual(kittiobj.score, 1.0)
    def test_equal(self):
        # same
        kittiobj1 = KittiObj('Pedestrian 0.0 0.0 -0.2 712.4 143.0 810.73 307.92 1.89 0.48 1.2 1.84 1.47 8.41 0.01')
        kittiobj2 = KittiObj('Pedestrian 0.0 0.0 -0.2 712.4 143.0 810.73 307.92 1.89 0.48 1.2 1.84 1.47 8.41 0.01')
        self.assertTrue(kittiobj1.equal(kittiobj1, 'Pedestrian', rtol=1e-5))
        self.assertTrue(kittiobj1.equal(kittiobj2, 'Pedestrian', rtol=1e-5))
        self.assertTrue(kittiobj2.equal(kittiobj1, 'Pedestrian', rtol=1e-5))
        self.assertTrue(kittiobj2.equal(kittiobj2, 'Pedestrian', rtol=1e-5))
        # ry vs. ry+np.pi
        kittiobj1 = KittiObj('Pedestrian 0.0 0.0 -0.2 712.4 143.0 810.73 307.92 1.89 0.48 1.2 1.84 1.47 8.41 0.01')
        kittiobj2 = KittiObj('Pedestrian 0.0 0.0 -0.2 712.4 143.0 810.73 307.92 1.89 0.48 1.2 1.84 1.47 8.41 3.1515926')
        self.assertTrue(kittiobj1.equal(kittiobj1, 'Pedestrian', rtol=1e-5))
        self.assertTrue(kittiobj1.equal(kittiobj2, 'Pedestrian', rtol=1e-5))
        self.assertTrue(kittiobj2.equal(kittiobj1, 'Pedestrian', rtol=1e-5))
        self.assertTrue(kittiobj2.equal(kittiobj2, 'Pedestrian', rtol=1e-5))
        # Different cls
        kittiobj1 = KittiObj('Pedestrian 0.0 0.0 -0.2 712.4 143.0 810.73 307.92 1.89 0.48 1.2 1.84 1.47 8.41 0.01')
        kittiobj2 = KittiObj('Car 0.0 0.0 -0.2 712.4 143.0 810.73 307.92 1.89 0.48 1.2 1.84 1.47 8.41 0.01')
        self.assertTrue(not kittiobj1.equal(kittiobj2, 'Pedestrian', rtol=1e-5))
        self.assertTrue(not kittiobj2.equal(kittiobj1, 'Pedestrian', rtol=1e-5))
        # Different ry
        kittiobj1 = KittiObj('Pedestrian 0.0 0.0 -0.2 712.4 143.0 810.73 307.92 1.89 0.48 1.2 1.84 1.47 8.41 0.01')
        kittiobj2 = KittiObj('Pedestrian 0.0 0.0 -0.2 712.4 143.0 810.73 307.92 1.89 0.48 1.2 1.84 1.47 8.41 1.31')
        self.assertTrue(not kittiobj1.equal(kittiobj2, 'Pedestrian', rtol=1e-5))
        self.assertTrue(not kittiobj2.equal(kittiobj1, 'Pedestrian', rtol=1e-5))        

class TestKittiLabel(unittest.TestCase):
    def test_init(self):
        kittilabel = KittiLabel('./unit-test/data/test_KittiLabel_000003.txt')
        self.assertEqual(kittilabel.data, None)
    def test_read_kitti_label_file(self):
        label = KittiLabel('./unit-test/data/test_KittiLabel_000003.txt').read_kitti_label_file(no_dontcare=True)
        self.assertEqual(len(label.data), 1)
        self.assertEqual(len(list(filter(lambda obj: obj.type == "DontCare", label.data))), 0)
        label = KittiLabel('./unit-test/data/test_KittiLabel_000003.txt').read_kitti_label_file(no_dontcare=False)
        self.assertEqual(len(label.data), 3)
        self.assertEqual(len(list(filter(lambda obj: obj.type == "DontCare", label.data))), 2)
    # TODO:TEST str
    def test_equal(self):
        label1 = KittiLabel('./unit-test/data/test_KittiLabel_000012.txt').read_kitti_label_file(no_dontcare=True)
        label2 = KittiLabel('./unit-test/data/test_KittiLabel_000012.txt').read_kitti_label_file(no_dontcare=True)
        self.assertTrue(label1.equal(label1, ['Car', 'Van'], rtol=1e-5))
        self.assertTrue(label1.equal(label2, ['Car', 'Van'], rtol=1e-5))
        self.assertTrue(label2.equal(label1, ['Car', 'Van'], rtol=1e-5))
        self.assertTrue(label2.equal(label2, ['Car', 'Van'], rtol=1e-5))
        label1 = KittiLabel('./unit-test/data/test_KittiLabel_000012.txt').read_kitti_label_file(no_dontcare=True)
        label2 = KittiLabel('./unit-test/data/test_KittiLabel_000003.txt').read_kitti_label_file(no_dontcare=True)
        self.assertTrue(not label1.equal(label2, ['Car', 'Van'], rtol=1e-5))
        self.assertTrue(not label2.equal(label1, ['Car', 'Van'], rtol=1e-5))

if __name__ == '__main__':
    unittest.main()
        