'''
File Created: Sunday, 30th June 2019 7:53:07 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
This script creates a post-processed dataset for KITTI and CARLA.
python3 methods/voxelnet/create_dataset.py \
    --data-dir /usr/app/data/KITTI \
    --dataset KITTI \
    --num-of-train 5
'''
import argparse
import os
import pickle
from det3.methods.voxelnet.config import cfg
def main():
    parser = argparse.ArgumentParser(description='This script creates a post-processed dataset for KITTI and CARLA.')
    parser.add_argument('--data-dir',
                        type=str, metavar='INPUT PATH',
                        help='dataset dir')
    parser.add_argument('--dataset',
                        type=str, metavar='dataset',
                        help='KITTI or CARLA')
    parser.add_argument('--num-of-train',
                        type=int, metavar='times of data augmentation for train dataset',
                        help='5')
    args = parser.parse_args()
    dataset = args.dataset
    data_dir = args.data_dir
    num_of_train = args.num_of_train
    cfg.bool_fast_loader = False # It is important, otherwise the fast load data will not be updated.
    if dataset == "KITTI":
        from det3.methods.voxelnet.kittidata import KittiDatasetVoxelNet
        for mode in ["dev", "val"]:
            try:
                print(mode)
                target_dir = os.path.join(data_dir, mode, "fast_load")
                os.makedirs(target_dir, exist_ok=False)
                dataloader = KittiDatasetVoxelNet(data_dir=data_dir, train_val_flag=mode, cfg=cfg)
                for _, data in enumerate(dataloader):
                    print("{:06d}".format(data[0]))
                    pickle.dump(data, open("{}/{:06d}.pkl".format(target_dir, data[0]), "wb"))
            except FileExistsError:
                print("FileExistsError RAISED: {} has already been created!".format(target_dir))

        mode = "train"
        for i in range(num_of_train):
            try:
                print(mode+str(i))
                target_dir = os.path.join(data_dir, mode, "fast_load_{}".format(i))
                os.makedirs(target_dir, exist_ok=False)
                dataloader = KittiDatasetVoxelNet(data_dir=data_dir, train_val_flag=mode, cfg=cfg)
                for _, data in enumerate(dataloader):
                    print("{:06d}".format(data[0]))
                    pickle.dump(data, open("{}/{:06d}.pkl".format(target_dir, data[0]), "wb"))
            except FileExistsError:
                print("FileExistsError RAISED: {} has already been created!".format(target_dir))

    elif dataset == "CARLA":
        from det3.methods.voxelnet.carladata import CarlaDatasetVoxelNet
        for mode in ["dev", "train", "val"]:
            print(mode)
            dataloader = CarlaDatasetVoxelNet(data_dir=data_dir, train_val_flag=mode, cfg=cfg)
            for _, data in enumerate(dataloader):
                print("{:06d}".format(data[0]))
                pickle.dump(data, open("{}/{}/{}/{:06d}.pkl".format(data_dir, mode, "fast_load", data[0]), "wb"))
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()