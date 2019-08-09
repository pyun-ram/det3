# det3
This is for RAM-Lab 3D object detector project.
# Requirements
- python 3.5.2
- pytorch 1.0.0+
- Cuda 9.0+ (If you want to use sparse conv)
# Docker
```
docker pull pyun/python_pytorch:cuda10_3.5_1.1
```
# TroubleShooting
```
# Trouble: ModuleNotFoundError: No module named 'det3'
# Solution: add <det3_rootdir>/../ to PYTHONPATH
# export PYTHONPATH=/root/second.pytorch/:/root/kitti-object-eval-python:<det3_rootdir>/../

# Trouble: ModuleNotFoundError: No module named 'dropblock'
# Solution:
# pip install dropblock

# Trouble: ModuleNotFoundError: No module named 'det3.methods.voxelnet.box_overlaps.box_overlaps'
# Solution:
# cd det3/
# ./methods/voxelnet/setup.sh
```