'''
File Created: Thursday, 18th July 2019 10:34:34 am
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
from det3.utils.torch_utils import GradientLogger, ActivationLogger
import os
from multiprocessing import Pool
from tqdm import tqdm

root_dir = "./methods/voxelnet/logs/VoxelNet-dev-test-biasinit/"
grad_data_dir = os.path.join(root_dir, "train_grad")
actv_data_dir = os.path.join(root_dir, "train_actv")
grad_logger = GradientLogger()
actv_logger = ActivationLogger()

def sort_fn(itm):
    return int(itm.split(".")[0])

def actv_fn(itm):
    actv_dict = actv_logger.load_pkl(os.path.join(actv_data_dir, itm))
    actv_logger.plot(actv_dict,
                     os.path.join(actv_data_dir, "{}.png".format(itm.split('.')[0])),
                     ylim=[-0.5, 0.5])
def grad_fn(itm):
    grad_dict = grad_logger.load_pkl(os.path.join(grad_data_dir, itm))
    grad_logger.plot(grad_dict,
                     os.path.join(grad_data_dir, "{}.png".format(itm.split('.')[0])),
                     ylim=[-0.001, 0.001])
def main():
    data_list = os.listdir(grad_data_dir)
    data_list = [itm for itm in data_list if itm.split(".")[-1] == "pkl"]
    data_list.sort(key=sort_fn)
    with Pool(8) as p:
        r = list(tqdm(p.imap(actv_fn, data_list), total=len(data_list)))
        r = list(tqdm(p.imap(grad_fn, data_list), total=len(data_list)))

if __name__ == "__main__":
    main()