import numpy as np
import pathlib
import copy
from functools import reduce
from det3.methods.second.data.mypreprocess import DBBatchSampler
from det3.utils.utils import read_pc_from_bin, apply_tr, apply_R, rotz
from det3.methods.second.utils.log_tool import Logger
class BaseDBSampler:
    def __init__(self):
        raise NotImplementedError

class DataBaseSamplerV3(BaseDBSampler):
    def __init__(self,
                 db_infos,
                 sample_dict:dict,
                 db_prepor=None,
                 sample_param:dict=None):
        for k, v in db_infos.items():
            Logger.log_txt(f"load {len(v)} {k} database infos")
        if db_prepor is not None:
            db_infos = db_prepor(db_infos)
            Logger.log_txt("After filter database:")
            for k, v in db_infos.items():
                Logger.log_txt(f"load {len(v)} {k} database infos")
        self._db_infos = db_infos
        self._sampler_dict = dict()
        for k, v in self._db_infos.items():
            self._sampler_dict[k] = DBBatchSampler(v, k, shuffle=True)
        self._sample_dict = sample_dict
        self._sample_param = sample_param

    def sample(self, gt_label, gt_calib):
        '''
        parameters:
            gt_label: KittiLabel of GT with current_frame == Cam2
            gt_calib: KittiCalib of GT
            sample_dict: {cls: max_num_objs} (deprecated)
                e.g. {
                    "Car": 15, # It will return the result containing maximum 15 cars (including GT)
                    ...
                }
        returns:
            res_label: KITTI label with current_frame == Cam2
            gt_mask : A mask corresponding to the res_label.data (If GT obj, true.)
            calib_list: a list of calib object corresponding to the res_label.data
            objpc_list: a list of object p.c. corresponding to the res_label.data
            Note that for the calib_list and objpc_list,
            the entries corresponding to GT obj will be None.
        '''
        from det3.dataloader.kittidata import Frame
        assert gt_label.current_frame == Frame.Cam2
        res_label = gt_label.copy()
        _sample_dict = self._sample_dict.copy()
        for cls in gt_label.bboxes_name:
            if cls in _sample_dict.keys():
                _sample_dict[cls] = _sample_dict[cls] - 1 if _sample_dict[cls] > 1 else 1
        for k, v in _sample_dict.items():
            _sample_dict[k] = np.random.randint(low=0, high=v, dtype=np.int16)
        gt_mask = [True] * len(gt_label.data)
        calib_list = [None] * len(gt_label.data)
        objpc_list = [None] * len(gt_label.data)
        for cls, num_sample in _sample_dict.items():
            if num_sample == 0:
                continue
            assert num_sample > 0, "num_sample must be positive integers"
            cls_samples = self._sampler_dict[cls].sample(num_sample)
            for cls_sample in cls_samples:
                attmps = 0
                # if origin mode (_set_location & _set_rotation), max_attmps should be 1
                max_attemps = 1 if self._sample_param["mode"] == "origin" else 10
                while attmps < max_attemps:
                    obj = cls_sample["box3d_cam"]
                    calib = cls_sample["calib"]
                    objpc = read_pc_from_bin(cls_sample["gtpc_path"])
                    self._set_location(obj, calib, objpc, self._sample_param)
                    self._set_rotation(obj, calib, objpc, self._sample_param)
                    if not self._have_collision(res_label, obj):
                        res_label.add_obj(obj)
                        gt_mask.append(False)
                        calib_list.append(calib)
                        objpc_list.append(objpc)
                        break
                    else:
                        attmps += 1
                        # if attmps >= max_attemps:
                        #     Logger.log_txt(f"DataBaseSamplerV3:sample: exceed "+
                        #                 f"max_attemps: {max_attemps}.")
        res_dict = {
            "res_label": res_label,
            "gt_mask": gt_mask,
            "calib_list": calib_list,
            "objpc_list": objpc_list,
            "num_obj": len(objpc_list)
        }
        return res_dict

    def _set_location(self, obj, calib, objpc, sample_param):
        mode = sample_param["mode"]
        if mode == "origin":
            pass
        elif mode == "random":
            attemp = 0
            max_attemps = 10
            while(attemp < max_attemps):
                x_range = sample_param["x_range"]
                y_range = sample_param["y_range"]
                z_range = sample_param["z_range"]
                tr_x = self._get_tr(x_range, obj, calib, "x")
                tr_y = self._get_tr(y_range, obj, calib, "y")
                tr_z = self._get_tr(z_range, obj, calib, "z")
                if (self._check_valid(obj, objpc, calib, [tr_x, tr_y, tr_z])):
                    self._apply_tr(obj, objpc, calib, [tr_x, tr_y, tr_z])
                    break
                else:
                    attemp += 1
                    # if attemp >= max_attemps:
                    #     Logger.log_txt(f"DataBaseSamplerV3:_set_location: exceed "+
                    #                 f"max_attemps: {max_attemps}.")

        else:
            raise NotImplementedError
        return

    def _check_valid(self, obj, objpc, calib, tr):
        bcenter_Fcam = np.array([obj.x, obj.y, obj.z]).reshape(1, -1)
        bcenter_Flidar = calib.leftcam2lidar(bcenter_Fcam)
        bcenter_Flidar = apply_tr(bcenter_Flidar, np.array(tr))
        mean_Flidar = apply_tr(objpc[:, :3].mean(axis=0).reshape(1, -1),
                               np.array(tr))
        dis = np.linalg.norm(mean_Flidar[:2] - bcenter_Flidar[:2], ord=2)
        x, y = bcenter_Flidar[0, :2]
        return dis < 5 and (y<x and -y<x)

    def _get_tr(self, xx_range, obj, calib, dim):
        if xx_range is None:
            return 0
        assert (isinstance(xx_range, tuple),
                "xx_range should be a None/('abs'/'rel', [min, max])")
        mode, minmax = xx_range
        _min, _max = minmax
        assert mode in ["abs", "rel"]
        assert dim in ["x", "y", "z"]
        dim2idx = {"x": 0, "y": 1, "z": 2}
        if mode == "abs":
            target_xx = float(np.random.uniform(_min, _max, 1))
            bcenter_Fcam = np.array([obj.x, obj.y, obj.z]).reshape(1,3)
            bcenter_Flidar = calib.leftcam2lidar(bcenter_Fcam)
            current_xx = bcenter_Flidar[0, dim2idx[dim]]
            rel_xx = target_xx - current_xx
        elif mode == "rel":
            rel_xx = float(np.random.uniform(_min, _max, 1))
        return rel_xx

    def _apply_tr(self, obj, objpc, calib, tr):
        # Note: After apply tr, the obj.bbox2d will not coincide with obj.bbox3d
        bcenter_Fcam = np.array([obj.x, obj.y, obj.z]).reshape(1, -1)
        bcenter_Flidar = calib.leftcam2lidar(bcenter_Fcam)
        bcenter_Flidar = apply_tr(bcenter_Flidar, np.array(tr))
        bcenter_Fcam = calib.lidar2leftcam(bcenter_Flidar)
        obj.x, obj.y, obj.z = bcenter_Fcam.flatten()

        objpc[:, :3] = apply_tr(objpc[:, :3], np.array(tr))

    def _set_rotation(self, obj, calib, objpc, sample_param):
        mode = sample_param["mode"]
        if mode == "origin":
            pass
        elif mode == "random":
            ry_range = sample_param["ry_range"]
            dry = self._get_dry(ry_range, obj, calib)
            self._apply_dry(obj, objpc, calib, dry)
        else:
            raise NotImplementedError
        return

    def _get_dry(self, ry_range, obj, calib):
        if ry_range is None:
            return 0
        assert (isinstance(ry_range, tuple),
                "ry_range should be a None/('abs'/'rel', [min, max])")
        mode, minmax = ry_range
        _min, _max = minmax
        assert mode in ["abs", "rel"]
        if mode == "abs":
            assert False, "Do not use abs range of ry for db sampling"
        elif mode == "rel":
            rel_ry = float(np.random.uniform(_min, _max, 1))
        return rel_ry

    def _apply_dry(self, obj, objpc, calib, dry):
        if dry == 0:
            return
        bottom_Fcam = np.array([obj.x, obj.y, obj.z]).reshape(1, -1)
        bottom_Flidar = calib.leftcam2lidar(bottom_Fcam)
        objpc[:, :3] = apply_tr(objpc[:, :3], -bottom_Flidar)
        # obj.ry += dry is correspond to rotz(-dry)
        # since obj is in cam frame
        # pc_ is in LiDAR frame
        objpc[:, :3] = apply_R(objpc[:, :3], rotz(-dry))
        objpc[:, :3] = apply_tr(objpc[:, :3], bottom_Flidar)
        # modify obj
        obj.ry += dry

    def _have_collision(self, label, new_obj):
        # TODO: Bottleneck in speed
        # (Specifically: KittiAugmentor check_overlap)
        from det3.dataloader.augmentor import KittiAugmentor
        _label = label.copy()
        _label.add_obj(new_obj)
        return not KittiAugmentor.check_overlap(label=_label)

class DataBaseSamplerV2(BaseDBSampler):
    from det3.methods.second.data.preprocess import (BatchSampler,
                                                     random_crop_frustum,
                                                     mask_points_in_corners,
                                                     noise_per_object_v3_,
                                                     box_collision_test)
    from det3.methods.second.utils.utils import shape_mergeable
    from det3.methods.second.ops.ops import (rotation_points_single_angle,
                                             box3d_to_bbox, center_to_corner_box2d,
                                             center_to_corner_box2d)
    def __init__(self,
                 db_infos,
                 groups,
                 db_prepor=None,
                 rate=1.0,
                 global_rot_range=None):
        for k, v in db_infos.items():
            print(f"load {len(v)} {k} database infos")
        if db_prepor is not None:
            db_infos = db_prepor(db_infos)
            print("After filter database:")
            for k, v in db_infos.items():
                print(f"load {len(v)} {k} database infos")

        self.db_infos = db_infos
        self._rate = rate
        self._groups = groups
        self._group_db_infos = {}
        self._group_name_to_names = []
        self._sample_classes = []
        self._sample_max_nums = []
        self._use_group_sampling = False  # slower
        if any([len(g) > 1 for g in groups]):
            self._use_group_sampling = True
        if not self._use_group_sampling:
            self._group_db_infos = self.db_infos  # just use db_infos
            for group_info in groups:
                group_names = list(group_info.keys())
                self._sample_classes += group_names
                self._sample_max_nums += list(group_info.values())
        else:
            raise NotImplementedError

        self._sampler_dict = {}
        for k, v in self._group_db_infos.items():
            self._sampler_dict[k] = BatchSampler(v, k)
        self._enable_global_rot = False
        if global_rot_range is not None:
            if not isinstance(global_rot_range, (list, tuple, np.ndarray)):
                global_rot_range = [-global_rot_range, global_rot_range]
            else:
                assert shape_mergeable(global_rot_range, [2])
            if np.abs(global_rot_range[0] - global_rot_range[1]) >= 1e-3:
                self._enable_global_rot = True
        self._global_rot_range = global_rot_range
        
    def sample_all(self,
                   root_path,
                   gt_boxes,
                   gt_names,
                   num_point_features,
                   random_crop=False,
                   gt_group_ids=None,
                   calib=None):
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(self._sample_classes,
                                              self._sample_max_nums):
            sampled_num = int(max_sample_num -
                              np.sum([n == class_name for n in gt_names]))
            sampled_num = np.round(self._rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled_groups = self._sample_classes
        if self._use_group_sampling:
            assert gt_group_ids is not None
            sampled_groups = []
            sample_num_per_class = []
            for group_name, class_names in self._group_name_to_names:
                sampled_nums_group = [sampled_num_dict[n] for n in class_names]
                sampled_num = np.max(sampled_nums_group)
                sample_num_per_class.append(sampled_num)
                sampled_groups.append(group_name)
            total_group_ids = gt_group_ids
        sampled = []
        sampled_gt_boxes = []
        avoid_coll_boxes = gt_boxes

        for class_name, sampled_num in zip(sampled_groups,
                                           sample_num_per_class):
            if sampled_num > 0:
                if self._use_group_sampling:
                    sampled_cls = self.sample_group(class_name, sampled_num,
                                                    avoid_coll_boxes,
                                                    total_group_ids)
                else:
                    sampled_cls = self.sample_class_v2(class_name, sampled_num,
                                                       avoid_coll_boxes)

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]["box3d_lidar"][
                            np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack(
                            [s["box3d_lidar"] for s in sampled_cls], axis=0)

                    sampled_gt_boxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0)
                    if self._use_group_sampling:
                        if len(sampled_cls) == 1:
                            sampled_group_ids = np.array(
                                sampled_cls[0]["group_id"])[np.newaxis, ...]
                        else:
                            sampled_group_ids = np.stack(
                                [s["group_id"] for s in sampled_cls], axis=0)
                        total_group_ids = np.concatenate(
                            [total_group_ids, sampled_group_ids], axis=0)

        if len(sampled) > 0:
            sampled_gt_boxes = np.concatenate(sampled_gt_boxes, axis=0)
            num_sampled = len(sampled)
            s_points_list = []
            for info in sampled:
                s_points = np.fromfile(
                    str(pathlib.Path(root_path) / info["path"]),
                    dtype=np.float32)
                s_points = s_points.reshape([-1, num_point_features])
                # if not add_rgb_to_points:
                #     s_points = s_points[:, :4]
                if "rot_transform" in info:
                    rot = info["rot_transform"]
                    s_points[:, :3] = rotation_points_single_angle(
                        s_points[:, :3], rot, axis=2)
                s_points[:, :3] += info["box3d_lidar"][:3]
                s_points_list.append(s_points)
                # print(pathlib.Path(info["path"]).stem)
            # gt_bboxes = np.stack([s["bbox"] for s in sampled], axis=0)
            # if np.random.choice([False, True], replace=False, p=[0.3, 0.7]):
            # do random crop.
            if random_crop:
                s_points_list_new = []
                assert calib is not None
                rect = calib["rect"]
                Trv2c = calib["Trv2c"]
                P2 = calib["P2"]
                gt_bboxes = box3d_to_bbox(sampled_gt_boxes, rect,
                                                     Trv2c, P2)
                crop_frustums = random_crop_frustum(
                    gt_bboxes, rect, Trv2c, P2)
                for i in range(crop_frustums.shape[0]):
                    s_points = s_points_list[i]
                    mask = mask_points_in_corners(
                        s_points, crop_frustums[i:i + 1]).reshape(-1)
                    num_remove = np.sum(mask)
                    if num_remove > 0 and (
                            s_points.shape[0] - num_remove) > 15:
                        s_points = s_points[np.logical_not(mask)]
                    s_points_list_new.append(s_points)
                s_points_list = s_points_list_new
            ret = {
                "gt_names": np.array([s["name"] for s in sampled]),
                "difficulty": np.array([s["difficulty"] for s in sampled]),
                "gt_boxes": sampled_gt_boxes,
                "points": np.concatenate(s_points_list, axis=0),
                "gt_masks": np.ones((num_sampled, ), dtype=np.bool_)
            }
            if self._use_group_sampling:
                ret["group_ids"] = np.array([s["group_id"] for s in sampled])
            else:
                ret["group_ids"] = np.arange(gt_boxes.shape[0],
                                             gt_boxes.shape[0] + len(sampled))
        else:
            ret = None
        return ret

    def sample(self, name, num):
        if self._use_group_sampling:
            group_name = name
            ret = self._sampler_dict[group_name].sample(num)
            groups_num = [len(l) for l in ret]
            return reduce(lambda x, y: x + y, ret), groups_num
        else:
            ret = self._sampler_dict[name].sample(num)
            return ret, np.ones((len(ret), ), dtype=np.int64)

    def sample_group(self, name, num, gt_boxes, gt_group_ids):
        sampled, group_num = self.sample(name, num)
        sampled = copy.deepcopy(sampled)
        # rewrite sampled group id to avoid duplicated with gt group ids
        gid_map = {}
        max_gt_gid = np.max(gt_group_ids)
        sampled_gid = max_gt_gid + 1
        for s in sampled:
            gid = s["group_id"]
            if gid in gid_map:
                s["group_id"] = gid_map[gid]
            else:
                gid_map[gid] = sampled_gid
                s["group_id"] = sampled_gid
                sampled_gid += 1

        num_gt = gt_boxes.shape[0]
        gt_boxes_bv = center_to_corner_box2d(
            gt_boxes[:, 0:2], gt_boxes[:, 3:5], gt_boxes[:, 6])

        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)
        sp_group_ids = np.stack([i["group_id"] for i in sampled], axis=0)
        valid_mask = np.zeros([gt_boxes.shape[0]], dtype=np.bool_)
        valid_mask = np.concatenate(
            [valid_mask,
             np.ones([sp_boxes.shape[0]], dtype=np.bool_)], axis=0)
        boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy()
        group_ids = np.concatenate([gt_group_ids, sp_group_ids], axis=0)
        if self._enable_global_rot:
            # place samples to any place in a circle.
            noise_per_object_v3_(
                boxes,
                None,
                valid_mask,
                0,
                0,
                self._global_rot_range,
                group_ids=group_ids,
                num_try=100)
        sp_boxes_new = boxes[gt_boxes.shape[0]:]
        sp_boxes_bv = center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6])
        total_bv = np.concatenate([gt_boxes_bv, sp_boxes_bv], axis=0)
        # coll_mat = collision_test_allbox(total_bv)
        coll_mat = box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False
        valid_samples = []
        idx = num_gt
        for num in group_num:
            if coll_mat[idx:idx + num].any():
                coll_mat[idx:idx + num] = False
                coll_mat[:, idx:idx + num] = False
            else:
                for i in range(num):
                    if self._enable_global_rot:
                        sampled[idx - num_gt +
                                i]["box3d_lidar"][:2] = boxes[idx + i, :2]
                        sampled[idx - num_gt +
                                i]["box3d_lidar"][6] = boxes[idx + i, 6]
                        sampled[idx - num_gt + i]["rot_transform"] = (
                            boxes[idx + i, 6] -
                            sp_boxes[idx + i - num_gt, 6])

                    valid_samples.append(sampled[idx - num_gt + i])
            idx += num
        return valid_samples

    def sample_class_v2(self, name, num, gt_boxes):
        sampled = self._sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_boxes.shape[0]
        num_sampled = len(sampled)
        gt_boxes_bv = center_to_corner_box2d(
            gt_boxes[:, 0:2], gt_boxes[:, 3:5], gt_boxes[:, 6])

        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)

        valid_mask = np.zeros([gt_boxes.shape[0]], dtype=np.bool_)
        valid_mask = np.concatenate(
            [valid_mask,
             np.ones([sp_boxes.shape[0]], dtype=np.bool_)], axis=0)
        boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy()
        if self._enable_global_rot:
            # place samples to any place in a circle.
            noise_per_object_v3_(
                boxes,
                None,
                valid_mask,
                0,
                0,
                self._global_rot_range,
                num_try=100)
        sp_boxes_new = boxes[gt_boxes.shape[0]:]
        sp_boxes_bv = center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6])

        total_bv = np.concatenate([gt_boxes_bv, sp_boxes_bv], axis=0)
        # coll_mat = collision_test_allbox(total_bv)
        coll_mat = box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                if self._enable_global_rot:
                    sampled[i - num_gt]["box3d_lidar"][:2] = boxes[i, :2]
                    sampled[i - num_gt]["box3d_lidar"][6] = boxes[i, 6]
                    sampled[i - num_gt]["rot_transform"] = (
                        boxes[i, 6] - sp_boxes[i - num_gt, 6])
                valid_samples.append(sampled[i - num_gt])
        return valid_samples

if __name__=="__main__":
    # from det3.methods.second.configs.config import cfg
    # from det3.methods.second.utils.import_tool import load_module
    # from det3.methods.second.data.preprocess import DataBasePreprocessor
    # import pickle
    # info_path = cfg.DataLoaders["DBSampler"]["db_info_path"]
    # dbprocers_cfg = cfg.DataLoaders["DBSampler"]["DBProcer"]
    # tmps = [(load_module("methods/second/data/preprocess.py", cfg["name"]), cfg) for cfg in dbprocers_cfg]
    # dbproces = []
    # for builder, params in tmps:
    #     del params["name"]
    #     dbproces.append(builder(**params))
    # db_prepor = DataBasePreprocessor(dbproces)
    # with open(info_path, "rb") as f:
    #     db_infos = pickle.load(f)

    from det3.methods.second.configs.config import cfg
    from det3.methods.second.utils.import_tool import load_module
    from det3.methods.second.data.mypreprocess import DataBasePreprocessor
    from det3.utils.utils import load_pickle
    from det3.dataloader.kittidata import KittiCalib, KittiLabel
    from det3.visualizer.vis import BEVImage, FVImage
    from PIL import Image
    import time
    from tqdm import tqdm

    np.random.seed(123)
    dbprocers_cfg = cfg.TrainDataLoader["DBSampler"]["DBProcer"]
    tmps = [(load_module("methods/second/data/mypreprocess.py", cfg["name"]), cfg) for cfg in dbprocers_cfg]
    dbproces = []
    for builder, params in tmps:
        del params["name"]
        dbproces.append(builder(**params))
    db_prepor = DataBasePreprocessor(dbproces)
    db_infos = load_pickle("/usr/app/data/MyKITTI/KITTI_dbinfos_train.pkl")
    train_infos = load_pickle("/usr/app/data/MyKITTI/KITTI_infos_train.pkl")
    sample_dict = cfg.TrainDataLoader["DBSampler"]["sample_dict"]
    sample_param = cfg.TrainDataLoader["DBSampler"]["sample_param"]
    dbsampler = DataBaseSamplerV3(db_infos,
                                  db_prepor=db_prepor,
                                  sample_dict=sample_dict,
                                  sample_param=sample_param)

    for j in range(len(train_infos)):
        train_info = train_infos[j]
        calib = train_info["calib"]
        label = train_info["label"]
        tag = train_info["tag"]
        print(tag)
        pc_reduced = read_pc_from_bin(train_info["reduced_pc_path"])
        orgbev = BEVImage(x_range=(0, 70), y_range=(-40, 40), grid_size=(0.1, 0.1))
        orgfv = FVImage()
        orgbev.from_lidar(pc_reduced)
        orgfv.from_lidar(calib, pc_reduced)
        for obj in label.data:
            orgbev.draw_box(obj, calib)
            orgfv.draw_3dbox(obj, calib)

        sample_res = dbsampler.sample(gt_label=label, gt_calib=calib)
        for i in range(sample_res["num_obj"]):
            obj = sample_res["res_label"].data[i]
            obj_calib = sample_res["calib_list"][i]
            objpc = sample_res["objpc_list"][i]
            if objpc is not None:
                # del origin pts in obj
                mask = obj.get_pts_idx(pc_reduced[:, :3], obj_calib)
                mask = np.logical_not(mask)
                pc_reduced = pc_reduced[mask, :]
                # add objpc
                pc_reduced = np.concatenate([pc_reduced, objpc], axis=0)
            else:
                pass # original obj

        resbev = BEVImage(x_range=(0, 70), y_range=(-40, 40), grid_size=(0.1, 0.1))
        resfv = FVImage()
        resbev.from_lidar(pc_reduced)
        resfv.from_lidar(calib, pc_reduced)
        for obj, obj_calib in zip(sample_res["res_label"].data, sample_res["calib_list"]):
            if obj_calib is None:
                resbev.draw_box(obj, calib)
                resfv.draw_3dbox(obj, calib)
            else:
                resbev.draw_box(obj, obj_calib)
                resfv.draw_3dbox(obj, obj_calib)

        orgbev_img = Image.fromarray(orgbev.data)
        orgbev_img.save(f"/usr/app/vis/b{tag}_orgbev_{sample_res['num_obj']}.png")
        resbev_img = Image.fromarray(resbev.data)
        resbev_img.save(f"/usr/app/vis/b{tag}_resbev_{sample_res['num_obj']}.png")
        orgfv_img = Image.fromarray(orgfv.data)
        orgfv_img.save(f"/usr/app/vis/f{tag}_orgfv_{sample_res['num_obj']}.png")
        resfv_img = Image.fromarray(resfv.data)
        resfv_img.save(f"/usr/app/vis/f{tag}_resfv_{sample_res['num_obj']}.png")
