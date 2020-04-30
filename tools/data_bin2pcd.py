'''
File Created: Monday, 8th April 2019 8:56:26 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
python3 tools/data_bin2pcd.py \
    --bin-dir /usr/app/mlod/tmp \
    --output-dir /usr/app/mlod/tmp_pcd
Note: It requires Open3D packages
'''
import argparse
import numpy as np
import os
import open3d as o3d

def read_pc_from_bin(bin_path):
    """Load PointCloud data from bin file. (KITTI Dataset for 4 channels)"""
    p = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return p

def save_pc_to_pcd(pc, pcd_path):
    '''
    save a frame of point cloud into a pcd file
    inputs:
        pc (numpy.array) [#pts, >=3]
        pcd_path (str)
        Note: Only xyz of pc can be saved
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:,:3])
    o3d.io.write_point_cloud(pcd_path, pcd)
    return

def get_list(dir_path):
    '''
    get the file list in dir
    '''
    return os.listdir(dir_path)

def bin2pcd(data_dir, output_dir):
    '''
    convert bin to pcd
    inputs:
        data_dir (str): 
            dir contains bin files
        output_dir (str): 
            dir saving pcd files
    returns:
        # of files
    '''
    assert os.path.isdir(data_dir)
    assert os.path.isdir(output_dir)
    bin_list = get_list(data_dir)
    for i, bin_path in enumerate(bin_list):
        print("{}/{}".format(i, len(bin_list)))
        tag = bin_path.split('/')[-1].split('.')[0]
        pc = read_pc_from_bin(os.path.join(data_dir, bin_path))
        save_pc_to_pcd(pc, os.path.join(output_dir, tag+'.pcd'))
    return len(bin_list)

def validate(bin_dir, pcd_dir):
    '''
    validate the result
    '''
    assert len(get_list(bin_dir)) == len(get_list(pcd_dir))

def main():
    '''
    convert bin into pcd
    '''
    parser = argparse.ArgumentParser(description='Split Dataset according to a txt file.')
    parser.add_argument('--bin-dir',
                        type=str, metavar='INPUT PATH',
                        help='dataset dir')
    parser.add_argument('--output-dir',
                        type=str, metavar='OUTPUT PATH',
                        help='output dir')
    args = parser.parse_args()
    data_dir = args.bin_dir
    output_dir = args.output_dir
    num_of_files = bin2pcd(data_dir, output_dir)
    print("Done: {}".format(num_of_files))


if __name__ == "__main__":
    main()
