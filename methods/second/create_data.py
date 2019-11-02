'''
File Created: Wednesday, 30th October 2019 2:51:52 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
Note: Create Data for 3D object detection. It creates info files and db files KITTI dataset.
Script: python3 methods/second/create_data.py \
            --dataset kitti \
            --data-dir /usr/app/data/MyKITTI/
Data organization: 
    KITTI/
        training/
            calib/
            image_2/
            velodyne/
            label_2/
            (veludyne_reduced/)
        testing/
            calib/
            image_2/
            velodyne/
            (veludyne_reduced/)
        split_index/
            train.txt
            val.txt
            test.txt
        (KITTI_infos_train.pkl)
        (KITTI_infos_val.pkl)
        (KITTI_infos_test.pkl)
        (gt_database/)
        (KITTI_dbinfos_train.pkl)
'''
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Manager, Pool
from det3.utils.utils import get_idx_list, read_pc_from_bin, save_pickle, write_pc_to_file
from det3.dataloader.kittidata import KittiData, KittiLabel

def reduce_pc(pc_Flidar, calib):
    '''
    reduce the point cloud and keep only the points in the cam2 frustum.
    '''
    width = np.ceil(calib.P2[0, 2] * 2).astype(np.int)
    height = np.ceil(calib.P2[1, 2] * 2).astype(np.int)
    front_mask = pc_Flidar[:, 0] > 0
    pc_Fcam = calib.lidar2leftcam(pc_Flidar[:, :3])
    pc_Fcam2d = calib.leftcam2imgplane(pc_Fcam[:, :3])
    mask1 = 0 < pc_Fcam2d[:, 0]
    mask2 = 0 < pc_Fcam2d[:, 1]
    mask3 = pc_Fcam2d[:, 0] < width
    mask4 = pc_Fcam2d[:, 1] < height
    mask = np.logical_and(front_mask, mask1)
    mask = np.logical_and(mask, mask2)
    mask = np.logical_and(mask, mask3)
    mask = np.logical_and(mask, mask4)
    return mask

def create_info_file_wk_fn(idx):
    global g_data_dir, g_infos
    root_dir = g_data_dir
    infos = g_infos
    output_dict = {"calib": True,
                   "image": False,
                   "label": True,
                   "velodyne": True}
    if root_dir.stem == "testing":
        output_dict['label'] = False # testing folder do not have label_2/ data.
    info = dict()
    tag = idx
    pc_path = str(root_dir/"velodyne"/f"{tag}.bin")
    reduced_pc_path = str(root_dir/"reduced_velodyne"/f"{tag}.bin")
    img_path = str(root_dir/"image_2"/f"{tag}.png")
    calib, _, label, pc_Flidar = KittiData(root_dir=str(root_dir), idx=tag,
                                           output_dict=output_dict).read_data()
    mask = reduce_pc(pc_Flidar, calib)
    pc_reduced_Flidar = pc_Flidar[mask, :]
    with open(reduced_pc_path, 'wb') as f:
        pc_reduced_Flidar.tofile(f)
    info["tag"] = tag
    info["pc_path"] = pc_path
    info["reduced_pc_path"] = reduced_pc_path
    info["img_path"] = img_path
    info["calib"] = calib
    info["label"] = label
    infos.append(info)

def create_info_file(root_dir:str, idx_path:str, save_path:str):
    '''
    Create KITTI_infos_xxx.pkl
    [<info>,]
    info: {
        tag: str (e.g. '000000'),
        pc_path: str,
        reduced_pc_path: str,
        img_path: str,
        calib: KittiCalib,
        label: KittiLable,
    }
    '''
    global g_data_dir, g_infos
    # load idx
    root_dir = Path(root_dir)
    (root_dir/"reduced_velodyne").mkdir(exist_ok=True)
    idx_list = get_idx_list(idx_path)
    idx_list.sort()
    infos = Manager().list()
    g_data_dir = root_dir
    g_infos = infos
    with Pool(8) as p:
        r = list(tqdm(p.imap(create_info_file_wk_fn, idx_list), total=len(idx_list)))
    infos = list(infos)
    save_pickle(infos, save_path)
    print(f"Created {save_path}: {len(infos)} samples")

def get_classes(label_dir:str, idx_list:list):
    '''
    Get all the classes by going through all the label files.
    '''
    res = []
    for idx in idx_list:
        label = KittiLabel(label_path=f"{label_dir}/{idx}.txt").read_label_file()
        for cls in label.bboxes_name:
            if cls not in res:
                res.append(cls)
    return res

def create_db_file_wk_fn(idx):
    global g_data_dir, g_dbinfos
    root_dir = g_data_dir
    tag = idx
    output_dict = {"calib": True,
                   "image": False,
                   "label": True,
                   "velodyne": True}
    calib, _, label, pc_Flidar = KittiData(root_dir=str(root_dir), idx=tag,
                                           output_dict=output_dict).read_data()
    for i, obj in enumerate(label.data):
        name = obj.type
        gt_idx = i
        gtpc_path = str(root_dir.parent/"gt_database"/f"{tag}_{name}_{gt_idx}.bin")
        gtpc_idx = obj.get_pts_idx(pc_Flidar[:, :3], calib)
        gtpc = pc_Flidar[gtpc_idx, :]
        box3d_cam = obj
        num_points_in_gt = gtpc.shape[0]
        write_pc_to_file(gtpc, gtpc_path)
        obj_info = {
            "name": name,
            "gtpc_path": gtpc_path,
            "tag": tag,
            "gt_idx": gt_idx,
            "box3d_cam": box3d_cam,
            "num_points_in_gt": num_points_in_gt,
            "calib": calib
            }
        g_dbinfos.append(obj_info)
        
def create_db_file(root_dir:str, idx_path:str, save_dir:str):
    '''
    Create KITTI_dbinfos_xxx.pkl and save gt_pc into gt_database/
    dbinfo: {
        "Car": [<car_dbinfo>, ],
        "Pedestrian": [<ped_dbinfo>, ], 
        ...
    }
    car_dbinfo:{
        name: str,
        gtpc_path: str,
        tag: str,
        gt_idx: int, # no. of obj
        box3d_cam: KittiObj,
        num_points_in_gt: int,
        calib: KittiCalib
    }
    '''
    global g_data_dir, g_dbinfos
    root_dir = Path(root_dir)
    save_dir = Path(save_dir)
    (save_dir/"gt_database").mkdir(exist_ok=True)
    idx_list = get_idx_list(idx_path)
    idx_list.sort()
    # get classes
    cls_list = get_classes(root_dir/"label_2", idx_list)
    dbinfo = {itm: [] for itm in cls_list}
    g_data_dir = root_dir
    g_dbinfos = Manager().list()

    with Pool(8) as p:
        r = list(tqdm(p.imap(create_db_file_wk_fn, idx_list), total=len(idx_list)))
    for info in g_dbinfos:
        dbinfo[info["name"]].append(info)
    save_pickle(dbinfo, str(save_dir/"KITTI_dbinfos_train.pkl"))
    print("KITTI_dbinfos_train.pkl saved.")
    for k, v in dbinfo.items():
        print(f"{k}: {len(v)}")

def main(args):
    global g_data_dir, g_infos, g_dbinfos
    dataset = args.dataset.lower()
    data_dir = Path(args.data_dir)
    assert dataset in ["kitti"], f"Sorry the {dataset} cannot be hundled."
    assert data_dir.exists()

    create_info_file(root_dir=str(data_dir/"training"),
                     idx_path=str(data_dir/"split_index"/"train.txt"),
                     save_path=str(data_dir/"KITTI_infos_train.pkl"))
    create_info_file(root_dir=str(data_dir/"training"),
                     idx_path=str(data_dir/"split_index"/"val.txt"),
                     save_path=str(data_dir/"KITTI_infos_val.pkl"))
    create_info_file(root_dir=str(data_dir/"testing"),
                     idx_path=str(data_dir/"split_index"/"test.txt"),
                     save_path=str(data_dir/"KITTI_infos_test.pkl"))
    create_db_file(root_dir=str(data_dir/"training"),
                   idx_path=str(data_dir/"split_index"/"train.txt"),
                   save_dir=str(data_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Data for KITTI dataset.')
    parser.add_argument('--dataset', type=str, help='kitti or carla')
    parser.add_argument('--data-dir', type=str, help='root dir of KITTI data')
    args = parser.parse_args()
    main(args)
