# VoxelNet
## Prerequisites
- Cython
## Compile
Follow the instruction of box_overlaps/setup.py
## VoxelNet++
If you want to use sparse convolution, you have to follow the following instructions.
### Install dependence python packages
```
pip install fire tensorboardX protobuf opencv-python numba scikit-image scipy pillow
```
Follow instructions in [spconv](https://github.com/pyun-ram/spconv) to install spconv.

If you want to train with fp16 mixed precision (train faster in RTX series, Titan V/RTX and Tesla V100, but I only have 1080Ti), you need to install [apex](https://github.com/pyun-ram/apex).

### Setup cuda for numba
you need to add following environment variable for numba.cuda, you can add them to ~/.bashrc:
```
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
```
