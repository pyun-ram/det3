'''
File Created: Tuesday, 19th March 2019 9:04:16 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import unittest
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
    def test_read_kitti_calib_file(self):
        calib = KittiCalib("./unit-test/data/test_KittiCalib_000000.txt").read_kitti_calib_file()
        self.assertTrue('P0' in calib.keys())
        self.assertTrue('P1' in calib.keys())
        self.assertTrue('P2' in calib.keys())
        self.assertTrue('P3' in calib.keys())
        self.assertTrue('R0_rect' in calib.keys())
        self.assertTrue('Tr_velo_to_cam' in calib.keys())
        self.assertTrue('Tr_imu_to_velo' in calib.keys())
        for k, v in calib.items():
            for itm in v:
                self.assertTrue(isinstance(itm, float))

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

class TestKittiLabel(unittest.TestCase):
    def test_init(self):
        kittilabel = KittiLabel('./unit-test/data/test_KittiLabel_000003.txt')
        self.assertEqual(kittilabel.data, None)
    def test_read_kitti_label_file(self):
        label = KittiLabel('./unit-test/data/test_KittiLabel_000003.txt').read_kitti_label_file(no_dontcare=True)
        self.assertEqual(len(label), 1)
        self.assertEqual(len(list(filter(lambda obj: obj.type == "DontCare", label))), 0)
        label = KittiLabel('./unit-test/data/test_KittiLabel_000003.txt').read_kitti_label_file(no_dontcare=False)
        self.assertEqual(len(label), 3)
        self.assertEqual(len(list(filter(lambda obj: obj.type == "DontCare", label))), 2)

if __name__ == '__main__':
    unittest.main()
        