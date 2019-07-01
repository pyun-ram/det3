'''
File Created: Sunday, 30th June 2019 7:53:07 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
This script creates a post-processed dataset for KITTI and CARLA.
python3 methods/voxelnet/create_dataset.py \
    --data-dir /usr/app/data/KITTI \
    --dataset KITTI
'''
import argparse
import sys
sys.path.append("../")
import os
import pickle

def main():
    parser = argparse.ArgumentParser(description='This script creates a post-processed dataset for KITTI and CARLA.')
    parser.add_argument('--data-dir',
                        type=str, metavar='INPUT PATH',
                        help='dataset dir')
    parser.add_argument('--dataset',
                        type=str, metavar='dataset',
                        help='KITTI or CARLA')
    args = parser.parse_args()
    dataset = args.dataset
    data_dir = args.data_dir
    os.makedirs(os.path.join(data_dir, "dev", "fast_load"), exist_ok=False)
    os.makedirs(os.path.join(data_dir, "val", "fast_load"), exist_ok=False)
    os.makedirs(os.path.join(data_dir, "train", "fast_load"), exist_ok=False)
    if dataset == "KITTI":
        from det3.methods.voxelnet.kittidata import KittiDatasetVoxelNet
        from det3.methods.voxelnet.config import cfg
        for mode in ["dev", "train", "val"]:
            print(mode)
            dataloader = KittiDatasetVoxelNet(data_dir=data_dir, train_val_flag=mode, cfg=cfg)
            for _, data in enumerate(dataloader):
                print("{:06d}".format(data[0]))
                pickle.dump(data, open("{}/{}/{}/{:06d}.pkl".format(data_dir, mode, "fast_load", data[0]), "wb"))
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()