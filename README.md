# det3

This is for the RAM-Lab 3D object detector project.
The architecture of the det3 project is as follows:
![arch](./figure/det3_v0.1_arch.png)

## Requirements

- python 3.7.3
- CUDA 10.1+
- PyTorch 1.4.0+
- Open3D 0.9


## Dockerfile

```sh
cd dockerfiles
docker build . -t <dockerimage-tag>/det3:v0.1
docker run -it --name det3 --gpus all <dockerimage-tag>/det3:v0.1
```
