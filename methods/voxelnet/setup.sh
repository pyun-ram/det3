#! /bin/bash
cd methods/voxelnet/box_overlaps/
export CFLAGS="-I /opt/conda/lib/python3.7/site-packages/numpy/core/include $CFLAGS"
python3 setup.py build_ext --inplace
cp ./det3/methods/voxelnet/box_overlaps/*.so ./