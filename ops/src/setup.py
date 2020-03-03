'''
 File Created: Mon Mar 02 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
'''

from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='iou_cpp',
      ext_modules=[cpp_extension.CppExtension('iou_cpp', ['iou.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
setup(name='boxop_cpp',
      ext_modules=[cpp_extension.CppExtension('boxop_cpp', ['boxop.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})