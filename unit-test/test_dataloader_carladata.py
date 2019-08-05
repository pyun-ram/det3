'''
File Created: Tuesday, 23rd April 2019 2:42:52 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import unittest
import numpy as np
try:
    from ..dataloarder.carladata import CarlaCalib, CarlaObj, CarlaLabel
except:
    # Run script
    from det3.dataloarder.carladata import CarlaCalib, CarlaObj, CarlaLabel

class TestCarlarCalib(unittest.TestCase):
    def test_init(self):
        calib = CarlaCalib("./unit-test/data/test_CarlaCalib_000000.txt")
        self.assertEqual(calib.data, None)
        self.assertEqual(calib.num_of_lidar, None)
        self.assertTrue(isinstance(calib.P0, np.ndarray))
    def test_read_calib_file(self):
        calib = CarlaCalib("./unit-test/data/test_CarlaCalib_000000.txt").read_calib_file()
        self.assertTrue('Tr_imu_to_velo_top' in calib.data.keys())
        self.assertTrue('Tr_imu_to_velo_left' in calib.data.keys())
        self.assertTrue('Tr_imu_to_velo_right' in calib.data.keys())
        for k, v in calib.data.items():
            for itm in v:
                self.assertTrue(isinstance(itm, float))
    def test_lidar2imu(self):
        calib = CarlaCalib("./unit-test/data/test_CarlaCalib_000000.txt").read_calib_file()
        pts = np.array([[ -6.80046272,   2.69413924,   1.59194338],
                        [ -6.77902794,   2.7376864 ,   1.59210932],
                        [-13.89153004,  33.10041809,   8.87361813]]
                        )
        results = np.array([[ -6.80046272,  2.69413924,  3.09194338],
                            [ -6.77902794,  2.7376864 ,  3.09210932],
                            [-13.89153004,  33.10041809, 10.37361813]])
        self.assertTrue(np.array_equal(results, calib.lidar2imu(pts, key='Tr_imu_to_velo_top')))
    def test_imu2cam(self):
        calib = CarlaCalib("./unit-test/data/test_CarlaCalib_000000.txt").read_calib_file()
        pts = np.array([[ -6.80046272,  2.69413924,  3.09194338],
                        [ -6.77902794,  2.7376864 ,  3.09210932],
                        [-13.89153004,  33.10041809, 10.37361813]])
        results = np.array([[ -2.69413924 , -3.09194338,  -6.80046272],
                            [ -2.7376864  , -3.09210932,  -6.77902794],
                            [-33.10041809 ,-10.37361813, -13.89153004]]
                            )
        self.assertTrue(np.allclose(results, calib.imu2cam(pts), rtol=1.e-5))
    def test_cam2imgplane(self):
        calib = CarlaCalib("./unit-test/data/test_CarlaCalib_000000.txt").read_calib_file()
        calib.P0 = np.array([[50, 0., 600, 0.],
                            [0., 50, 180, 0.],
                            [0., 0., 1, 0.]])
        pts = np.array([[ -2.69413924 , -3.09194338,  -6.80046272],
                        [ -2.7376864  , -3.09210932,  -6.77902794],
                        [-33.10041809 ,-10.37361813, -13.89153004]]
                        )
        results = np.array([[619.80849945, 202.73333087],
                            [620.19232271, 202.8064359 ],
                            [719.13884937, 217.33792497]]
                            )
        self.assertTrue(np.allclose(results, calib.cam2imgplane(pts), rtol=1.e-5))
    # TODO: test_cam2imu

class TestCarlaObj(unittest.TestCase):
    def test_init(self):
        obj = CarlaObj('Car 0.00 0 1.55 614.24 181.78 727.31 284.77 1.57 1.73 4.15 1.00 1.75 13.22 1.62')
        self.assertEqual(obj.type, 'Car')
        self.assertEqual(obj.truncated, 0)
        self.assertEqual(obj.occluded, 0)
        self.assertEqual(obj.alpha, 1.55)
        self.assertEqual(obj.bbox_l, 614.24)
        self.assertEqual(obj.bbox_t, 181.78)
        self.assertEqual(obj.bbox_r, 727.31)
        self.assertEqual(obj.bbox_b, 284.77)
        self.assertEqual(obj.h, 1.57)
        self.assertEqual(obj.w, 1.73)
        self.assertEqual(obj.l, 4.15)
        self.assertEqual(obj.x, 1.00)
        self.assertEqual(obj.y, 1.75)
        self.assertEqual(obj.z, 13.22)
        self.assertEqual(obj.ry, 1.62)
        self.assertEqual(obj.score, None)
        obj = CarlaObj('Car 0.00 0 1.55 614.24 181.78 727.31 284.77 1.57 1.73 4.15 1.00 1.75 13.22 1.62 0.99')
        self.assertEqual(obj.type, 'Car')
        self.assertEqual(obj.truncated, 0)
        self.assertEqual(obj.occluded, 0)
        self.assertEqual(obj.alpha, 1.55)
        self.assertEqual(obj.bbox_l, 614.24)
        self.assertEqual(obj.bbox_t, 181.78)
        self.assertEqual(obj.bbox_r, 727.31)
        self.assertEqual(obj.bbox_b, 284.77)
        self.assertEqual(obj.h, 1.57)
        self.assertEqual(obj.w, 1.73)
        self.assertEqual(obj.l, 4.15)
        self.assertEqual(obj.x, 1.00)
        self.assertEqual(obj.y, 1.75)
        self.assertEqual(obj.z, 13.22)
        self.assertEqual(obj.ry, 1.62)
        self.assertEqual(obj.score, 0.99)
    def test_equal(self):
        # same
        obj1 = CarlaObj('Pedestrian 0.0 0.0 -0.2 712.4 143.0 810.73 307.92 1.89 0.48 1.2 1.84 1.47 8.41 0.01')
        obj2 = CarlaObj('Pedestrian 0.0 0.0 -0.2 712.4 143.0 810.73 307.92 1.89 0.48 1.2 1.84 1.47 8.41 0.01')
        self.assertTrue(obj1.equal(obj1, 'Pedestrian', rtol=1e-5))
        self.assertTrue(obj1.equal(obj2, 'Pedestrian', rtol=1e-5))
        self.assertTrue(obj2.equal(obj1, 'Pedestrian', rtol=1e-5))
        self.assertTrue(obj2.equal(obj2, 'Pedestrian', rtol=1e-5))
        # ry vs. ry+np.pi
        obj1 = CarlaObj('Pedestrian 0.0 0.0 -0.2 712.4 143.0 810.73 307.92 1.89 0.48 1.2 1.84 1.47 8.41 0.01')
        obj2 = CarlaObj('Pedestrian 0.0 0.0 -0.2 712.4 143.0 810.73 307.92 1.89 0.48 1.2 1.84 1.47 8.41 3.1515926')
        self.assertTrue(obj1.equal(obj1, 'Pedestrian', rtol=1e-5))
        self.assertTrue(obj1.equal(obj2, 'Pedestrian', rtol=1e-5))
        self.assertTrue(obj2.equal(obj1, 'Pedestrian', rtol=1e-5))
        self.assertTrue(obj2.equal(obj2, 'Pedestrian', rtol=1e-5))
        # Different cls
        obj1 = CarlaObj('Pedestrian 0.0 0.0 -0.2 712.4 143.0 810.73 307.92 1.89 0.48 1.2 1.84 1.47 8.41 0.01')
        obj2 = CarlaObj('Car 0.0 0.0 -0.2 712.4 143.0 810.73 307.92 1.89 0.48 1.2 1.84 1.47 8.41 0.01')
        self.assertTrue(not obj1.equal(obj2, 'Pedestrian', rtol=1e-5))
        self.assertTrue(not obj2.equal(obj1, 'Pedestrian', rtol=1e-5))
        # Different ry
        obj1 = CarlaObj('Pedestrian 0.0 0.0 -0.2 712.4 143.0 810.73 307.92 1.89 0.48 1.2 1.84 1.47 8.41 0.01')
        obj2 = CarlaObj('Pedestrian 0.0 0.0 -0.2 712.4 143.0 810.73 307.92 1.89 0.48 1.2 1.84 1.47 8.41 1.31')
        self.assertTrue(not obj1.equal(obj2, 'Pedestrian', rtol=1e-5))
        self.assertTrue(not obj2.equal(obj1, 'Pedestrian', rtol=1e-5))
    def test_getbbox3dcorners(self):
        obj = CarlaObj('Car 0.0 0.0 0.0 0.000 0.000 0.000 0.000 0.743 0.766 1.336 -23.748 -84.773 1.120 -1.569 0.990')
        res = np.array([[-23.36589923, -84.10430271,   1.11981429],
                        [-23.36354509, -85.4401231,    1.11981429],
                        [-24.12977693, -85.44147344,   1.11981429],
                        [-24.13213107, -84.10565306,   1.11981429],
                        [-23.36589923, -84.10430271,   1.86283435],
                        [-23.36354509, -85.4401231,    1.86283435],
                        [-24.12977693, -85.44147344,   1.86283435],
                        [-24.13213107, -84.10565306,   1.86283435]])
        self.assertTrue(np.allclose(res, obj.get_bbox3dcorners(), rtol=1e-2))
    # TODO: test_getpts
    # TODO: test_fromcorners
class TestCarlarLabel(unittest.TestCase):
    def test_init(self):
        label = CarlaLabel('./unit-test/data/test_CarlaLabel_000000.txt')
        self.assertEqual(label.data, None)
    def test_read_label_file(self):
        label = CarlaLabel('./unit-test/data/test_CarlaLabel_000000.txt').read_label_file(no_dontcare=True)
        self.assertEqual(len(label.data), 60)
    # TODO:TEST str
    def test_equal(self):
        label1 = CarlaLabel('./unit-test/data/test_CarlaLabel_000000.txt').read_label_file(no_dontcare=True)
        label2 = CarlaLabel('./unit-test/data/test_CarlaLabel_000000.txt').read_label_file(no_dontcare=True)
        self.assertTrue(label1.equal(label1, ['Car', 'Van'], rtol=1e-5))
        self.assertTrue(label1.equal(label2, ['Car', 'Van'], rtol=1e-5))
        self.assertTrue(label2.equal(label1, ['Car', 'Van'], rtol=1e-5))
        self.assertTrue(label2.equal(label2, ['Car', 'Van'], rtol=1e-5))
        label1 = CarlaLabel('./unit-test/data/test_CarlaLabel_000000.txt').read_label_file(no_dontcare=True)
        label2 = CarlaLabel('./unit-test/data/test_CarlaLabel_000010.txt').read_label_file(no_dontcare=True)
        self.assertTrue(not label1.equal(label2, ['Car', 'Van'], rtol=1e-5))
        self.assertTrue(not label2.equal(label1, ['Car', 'Van'], rtol=1e-5))

if __name__ == '__main__':
    unittest.main()
