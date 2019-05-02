#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : setup.py
# Purpose :
# Creation Date : 11-12-2017
# Last Modified : Sat 23 Dec 2017 03:18:37 PM CST
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]
# Thanks to Jeasine Ma! :)
# In the det3/methods/voxelnet/box_overlaps/: RUN python3 setup.py build_ext --inplace
# cd det3/methods/voxelnet/box_overlaps/
# python3 setup.py build_ext --inplace
# mv det3/methods/voxelnet/box_overlaps/det3/methods/voxelnet/box_overlaps/*.so det3/methods/voxelnet/box_overlaps
# rm det3/methods/voxelnet/box_overlaps/det3/methods/voxelnet/box_overlaps/

from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='box overlaps',
    ext_modules=cythonize('box_overlaps.pyx')
)

# solution for potential error related to numpy/arrayobject.h
# export CFLAGS="-I /usr/local/lib/python3.5/dist-packages/numpy/core/include $CFLAGS"
