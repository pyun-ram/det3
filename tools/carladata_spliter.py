'''
File Created: Sunday, 17th March 2019 11:18:42 am
Author: Peng YUN (pyun@ust.hk) Jianhao JIAO (jjiao@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
Usage: python3 tools/carladata_spliter.py \
    --data-dir /usr/app/data/CARLA/training \
    --idx-file /usr/app/data/CARLA/split_index/dev.txt \
    --output-dir /usr/app/data/CARLA/dev
'''
import argparse
import os
from det3.utils.utils import get_idx_list

def check_datadir_valid(data_dir):
    '''
    check if data_dir is valid:
        - if exist
        - if contains calib, image_2, label_2, velodyne
        - if the #s of files in the above four dir are the same
    inputs:
        data_dir(str): the dataset dir
    '''
    assert os.path.exists(data_dir)
    assert os.path.exists(os.path.join(data_dir, "calib"))
    # assert os.path.exists(os.path.join(data_dir, "image_2"))
    assert os.path.exists(os.path.join(data_dir, "label_imu"))
    assert os.path.exists(os.path.join(data_dir, "velo_top"))
    assert os.path.exists(os.path.join(data_dir, "velo_left"))
    assert os.path.exists(os.path.join(data_dir, "velo_right"))
    assert len(os.listdir(os.path.join(data_dir, "calib"))) == \
        len(os.listdir(os.path.join(data_dir, "label_imu"))) == \
        len(os.listdir(os.path.join(data_dir, "velo_top"))) == \
        len(os.listdir(os.path.join(data_dir, "velo_left"))) == \
        len(os.listdir(os.path.join(data_dir, "velo_right")))

def check_idxfile_valid(idx_path):
    '''
    check if idx_path is valid:
        - if exist
    inputs:
        idx_path(str): the idx_path
    '''
    assert os.path.exists(idx_path)


def split_data(data_dir, idx_path, output_dir):
    '''
    split dataset according to the index file.
    inputs:
        data_dir(str): the dataset dir
        idx_path(str): the idx_path
        output_dir(str): the output dir
    outputs:
        num_data(int): the # of samples should be in the output_dir
    '''
    # get idx_file
    idx_list = get_idx_list(idx_path)
    os.mkdir(output_dir)
    os.mkdir(os.path.join(output_dir, 'calib'))
    # os.mkdir(os.path.join(output_dir, 'image_2'))
    os.mkdir(os.path.join(output_dir, 'label_imu'))
    os.mkdir(os.path.join(output_dir, 'velo_top'))
    os.mkdir(os.path.join(output_dir, 'velo_left'))
    os.mkdir(os.path.join(output_dir, 'velo_right'))

    # symlink data
    idx_count = 0
    for idx in idx_list:
        if not os.path.isfile(os.path.join(data_dir, 'calib', idx+'.txt')):
            return idx_count

        idx_count = idx_count + 1
        os.symlink(os.path.join(data_dir, 'calib', idx+'.txt'), os.path.join(output_dir, 'calib', idx+'.txt'))
        # os.symlink(os.path.join(data_dir, 'image_2', idx+'.png'), os.path.join(output_dir, 'image_2', idx+'.png'))
        os.symlink(os.path.join(data_dir, 'label_imu', idx+'.txt'), os.path.join(output_dir, 'label_imu', idx+'.txt'))
        os.symlink(os.path.join(data_dir, 'velo_top', idx+'.npy'), os.path.join(output_dir, 'velo_top', idx+'.npy'))
        os.symlink(os.path.join(data_dir, 'velo_left', idx + '.npy'), os.path.join(output_dir, 'velo_left', idx + '.npy'))
        os.symlink(os.path.join(data_dir, 'velo_right', idx + '.npy'), os.path.join(output_dir, 'velo_right', idx + '.npy'))
    return len(idx_list)

def validate(output_dir, num_data):
    '''
    validate if the sampling is correct.
    inputs:
        output_dir(str): the output dir
        num_data(int): the # of samples should be in the output_dir
    '''
    assert num_data == len(os.listdir(os.path.join(output_dir, "calib")))
    # assert num_data == len(os.listdir(os.path.join(output_dir, "image_2")))
    assert num_data == len(os.listdir(os.path.join(output_dir, "label_imu")))
    assert num_data == len(os.listdir(os.path.join(output_dir, "velo_top")))
    assert num_data == len(os.listdir(os.path.join(output_dir, "velo_left")))
    assert num_data == len(os.listdir(os.path.join(output_dir, "velo_right")))
    print("# of data: {}".format(num_data))
    print("{}: DONE".format(__file__))

def main():
    '''
    split data
    '''
    # print('example: python3 tools/carla_data_spliter.py --data-dir /usr/app/data/CARLA/raw_data/ --idx-file ../../data/CARLA/split_index/train.txt --output-dir ../../data/CARLA/train/')
    parser = argparse.ArgumentParser(description='Split Dataset according to a txt file.')
    parser.add_argument('--data-dir',
                        type=str, metavar='INPUT PATH',
                        help='dataset dir')
    parser.add_argument('--idx-file',
                        type=str, metavar='INDEX FILE PATH',
                        help='the txt file containing the indeces of the smapled data')
    parser.add_argument('--output-dir',
                        type=str, metavar='OUTPUT PATH',
                        help='output dir')
    args = parser.parse_args()
    data_dir = args.data_dir
    idx_path = args.idx_file
    output_dir = args.output_dir
    check_datadir_valid(data_dir)
    check_idxfile_valid(idx_path)
    num_data = split_data(data_dir, idx_path, output_dir)
    # report the result
    validate(output_dir, num_data)

if __name__ == "__main__":
    main()