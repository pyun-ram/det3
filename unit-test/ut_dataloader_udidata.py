'''
 File Created: Thu Mar 19 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
'''
import time
import torch
import unittest
import numpy as np
from det3.dataloader.udidata import UdiFrame

class UTUDIFRAME(unittest.TestCase):
    def test(self):
        self.assertTrue(UdiFrame.all_frames() == ", ".join(UdiFrame.Frame._member_names_))

if __name__ == "__main__":
    unittest.main()
