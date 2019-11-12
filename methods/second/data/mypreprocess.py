import numpy as np
import abc
from det3.dataloader.augmentor import KittiAugmentor
from det3.dataloader.kittidata import KittiLabel, Frame
from det3.methods.second.utils.utils import filt_label_by_cls

class DBBatchSampler:
    def __init__(self, sample_list, name, shuffle):
        "Source: https://github.com/traveller59/second.pytorch"
        self._sample_list = sample_list
        self._indices = np.arange(len(sample_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sample_list)
        self._name = name
        self._shuffle = shuffle

    def _sample(self, num):
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx:].copy()
            self._reset()
        else:
            ret = self._indices[self._idx:self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        if self._name is not None:
            print("reset", self._name)
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
    
    def sample(self, num):
        indices = self._sample(num)
        return [self._sample_list[i] for i in indices]

class DataBasePreprocessing:
    def __call__(self, db_infos):
        return self._preprocess(db_infos)

    @abc.abstractclassmethod
    def _preprocess(self, db_infos):
        pass

class DBFilterByMinNumPoint(DataBasePreprocessing):
    def __init__(self, min_gt_point_dict):
        self._min_gt_point_dict = min_gt_point_dict
        print(min_gt_point_dict)

    def _preprocess(self, db_infos):
        for name, min_num in self._min_gt_point_dict.items():
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info["num_points_in_gt"] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos

class DataBasePreprocessor:
    def __init__(self, preprocessors):
        self._preprocessors = preprocessors

    def __call__(self, db_infos):
        for prepor in self._preprocessors:
            db_infos = prepor(db_infos)
        return db_infos

def prep_pointcloud(input_dict,
                    training,
                    db_sampler,
                    augment_dict,
                    voxelizer,
                    max_voxels,
                    anchor_cache,
                    target_assigner):
    pc_reduced = input_dict["lidar"]["points"]
    gt_calib = input_dict["calib"]
    tag = input_dict["metadata"]["tag"]

    if training:
        gt_label = input_dict["cam"]["label"]
        # db sampling
        if db_sampler is not None:
            sample_res = db_sampler.sample(gt_label=gt_label, gt_calib=gt_calib)
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
            np.random.shuffle(pc_reduced)
            calib_list = [itm if itm is not None else gt_calib for itm in sample_res["calib_list"]]
            label = sample_res["res_label"]
        # augmentation
        augmentor = KittiAugmentor(**augment_dict)
        label, pc_reduced = augmentor.apply(label, pc_reduced, calib_list)
        # label cleaning
        label = filt_label_by_cls(label, keep_cls=target_assigner.classes)
    else:
        label = input_dict["cam"]["label"]
        pc_reduced = pc_reduced
        calib_list = [gt_calib] * len(label.data)
    # Voxelization
    vox_res = voxelizer.generate(pc_reduced, max_voxels)
    voxels = vox_res["voxels"]
    coordinates = vox_res["coordinates"]
    num_points = vox_res["num_points_per_voxel"]
    num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
    example = {
        'tag': tag,
        'voxels': voxels,
        'num_points': num_points,
        'coordinates': coordinates,
        "num_voxels": num_voxels,
    }
    # Create Anchors
    if anchor_cache is not None:
        anchors = anchor_cache["anchors"]
        anchors_bv = anchor_cache["anchors_bv"]
        anchors_dict = anchor_cache["anchors_dict"]
        matched_thresholds = anchor_cache["matched_thresholds"]
        unmatched_thresholds = anchor_cache["unmatched_thresholds"]

    else:
        from det3.methods.second.ops.ops import rbbox2d_to_near_bbox
        grid_size = voxelizer.grid_size
        out_size_factor=2
        feature_map_size = grid_size[:2] // out_size_factor
        feature_map_size = [*feature_map_size, 1][::-1]
        ret = target_assigner.generate_anchors(feature_map_size)
        anchors = ret["anchors"]
        anchors = anchors.reshape([-1, target_assigner.box_ndim])
        anchors_dict = target_assigner.generate_anchors_dict(feature_map_size)
        anchors_bv = rbbox2d_to_near_bbox(
            anchors[:, [0, 1, 3, 4, 6]])
        matched_thresholds = ret["matched_thresholds"]
        unmatched_thresholds = ret["unmatched_thresholds"]
    example["anchors"] = anchors
    anchors_mask = None
    if not training:
        return example
    # Target Assignment
    class_names = target_assigner.classes
    gt_dict = kittilabel2gt_dict(label, class_names, calib_list)
    targets_dict = target_assigner.assign(
        anchors,
        anchors_dict,
        gt_dict["gt_boxes"],
        anchors_mask,
        gt_classes=gt_dict["gt_classes"],
        gt_names=gt_dict["gt_names"],
        matched_thresholds=matched_thresholds,
        unmatched_thresholds=unmatched_thresholds,
        importance=gt_dict["gt_importance"])
    example.update({
        'labels': targets_dict['labels'],
        'reg_targets': targets_dict['bbox_targets'],
        # 'reg_weights': targets_dict['bbox_outside_weights'],
        'importance': targets_dict['importance'],
    })
    return example

def kittilabel2gt_dict(kittilabel: KittiLabel, class_names, calib_list) -> dict:
    assert kittilabel.current_frame == Frame.Cam2
    if not (kittilabel.data is None or kittilabel.data == []):
        gt_boxes = kittilabel.bboxes3d
        wlh = gt_boxes[:, [1, 2, 0]]
        ry = gt_boxes[:, -1:]
        xyz_Flidar = np.zeros_like(wlh)
        for i, (obj, calib) in enumerate(zip(kittilabel.data, calib_list)):
            bcenter_Fcam = np.array([obj.x, obj.y, obj.z]).reshape(1, 3)
            bcenter_Flidar = calib.leftcam2lidar(bcenter_Fcam)
            xyz_Flidar[i, :] = bcenter_Flidar
        gt_boxes_Flidar = np.concatenate([xyz_Flidar, wlh, ry], axis=1)
        gt_names = kittilabel.bboxes_name
        gt_classes = np.array(
            [class_names.index(n) + 1 for n in gt_names],
            dtype=np.int32)
        gt_importance = np.ones([len(kittilabel.data)], dtype=gt_boxes.dtype)
    else:
        gt_boxes_Flidar = np.array([])
        gt_classes = []
        gt_names = []
        gt_importance = []
    gt_dict = {
        "gt_boxes": gt_boxes_Flidar,
        "gt_classes": gt_classes,
        "gt_names": gt_names,
        "gt_importance": gt_importance
    }
    return gt_dict
