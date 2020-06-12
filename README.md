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

## Results:

SECOND:
```
# Single class
Car.bbox@0.70: 84.98, 83.04, 75.32
Car.bev@0.70:  89.46, 86.48, 79.17
Car.3d@0.70:   86.95, 76.42, 74.94

Car.bev@0.50:  90.65, 89.67, 88.89
Car.3d@0.50:   90.65, 89.62, 88.73

# Multi Class
Car.bbox@0.70:      85.13, 81.99, 75.17
Car.bev@0.70:       89.84, 85.97, 79.22
Car.3d@0.70:        86.52, 75.22, 68.55
Car.bev@0.50:       90.57, 89.27, 88.34
Car.3d@0.50:        90.57, 89.18, 88.11

Pedestrian.bbox@0.5: 72.0, 64.54, 61.58
Pedestrian.bev@0.50: 66.86, 59.09, 52.91
Pedestrian.3d@0.5:   59.0, 51.54, 44.92
Pedestrian.bev@0.25: 81.19, 74.11, 70.82
Pedestrian.3d@0.25:  81.15, 73.96, 70.65

Cyclist.bbox@0.50:   82.26, 64.92, 63.41
Cyclist.bev@0.50:    78.28, 60.23, 54.46
Cyclist.3d@0.50:     75.42, 58.32, 52.75
Cyclist.bev@0.25:    81.6, 68.35, 62.14
Cyclist.3d@0.25:     81.59, 63.54, 62.01

Van.bbox@0.7:        28.52, 27.3, 27.64
Van.bev@0.70:        36.75, 32.55, 29.94
Van.3d@0.70:         26.88, 25.89, 23.58
Van.bev@0.5:         37.07, 33.1, 30.82
Van.3d@0.50:         37.06, 33.02, 30.72
```
