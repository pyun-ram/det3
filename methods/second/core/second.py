import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from det3.methods.second.models import voxel_encoder, middle, rpn, losses
from det3.methods.second.utils import metrics
from det3.methods.second.utils.log_tool import Logger
from det3.methods.second.utils.torch_utils import one_hot
from det3.methods.second.ops.ops import limit_period, center_to_corner_box2d, corner_to_standup_nd
from det3.methods.second.ops.torch_ops import rotate_nms

def get_downsample_factor(model_config):
    downsample_factor = np.prod(model_config["RPN"]["layer_strides"])
    if len(model_config["RPN"]["upsample_strides"]) > 0:
        downsample_factor /= model_config["RPN"]["upsample_strides"][-1]
    downsample_factor *= model_config["MiddleLayer"]["downsample_factor"]
    downsample_factor = np.round(downsample_factor).astype(np.int64)
    assert downsample_factor > 0
    return downsample_factor

class VoxelNet(nn.Module):
    def __init__(self,
                 vfe_cfg:dict,
                 middle_cfg:dict,
                 rpn_cfg:dict,
                 cls_loss_cfg:dict,
                 loc_loss_cfg:dict,
                 cfg: dict,
                 target_assigner,
                 voxelizer,
                 name="VoxelNet"):
        super().__init__()
        super().__init__()
        self.name = name
        self._sin_error_factor = cfg["sin_error_factor"]
        self._num_class = cfg["num_class"]
        self._use_rotate_nms = cfg["use_rotate_nms"]
        self._multiclass_nms = cfg["multiclass_nms"]
        self._nms_score_thresholds = cfg["nms_score_thresholds"]
        self._nms_pre_max_sizes = cfg["nms_pre_max_sizes"]
        self._nms_post_max_sizes = cfg["nms_post_max_sizes"]
        self._nms_iou_thresholds = cfg["nms_iou_thresholds"]
        self._use_sigmoid_score = cfg["use_sigmoid_score"]
        self._encode_background_as_zeros = cfg["encode_background_as_zeros"]
        self._use_direction_classifier = cfg["use_direction_classifier"]
        self._num_input_features = cfg["num_input_features"]
        self._box_coder = target_assigner.box_coder
        self.target_assigner = target_assigner
        self.voxel_generator = voxelizer
        self._pos_cls_weight = cfg["pos_cls_weight"]
        self._neg_cls_weight = cfg["neg_cls_weight"]
        self._encode_rad_error_by_sin = cfg["encode_rad_error_by_sin"]
        self._loss_norm_type = cfg["loss_norm_type"]
        self._dir_loss_ftor = losses.WeightedSoftmaxClassificationLoss() #TODO: Change to Configable
        self._diff_loc_loss_ftor = losses.WeightedSmoothL1LocalizationLoss() #TODO: Change to Configable
        Logger.log_txt("Warning: _dir_loss_ftor and _diff_loc_loss_ftor need to be changed to configable.")
        self._dir_offset = cfg["direction_offset"]
        self._cls_loss_ftor = losses.get_loss_class(cls_loss_cfg["name"])(**cls_loss_cfg)
        self._loc_loss_ftor = losses.get_loss_class(loc_loss_cfg["name"])(**loc_loss_cfg)
        self._direction_loss_weight = cfg["direction_loss_weight"]
        self._cls_loss_weight = cfg["cls_loss_weight"]
        self._loc_loss_weight = cfg["loc_loss_weight"]
        self._post_center_range = cfg["post_center_range"] or []
        self._nms_class_agnostic = cfg["nms_class_agnostic"]
        self._num_direction_bins = cfg["num_direction_bins"]
        self._dir_limit_offset = cfg["direction_limit_offset"]

        rpn_cfg["encode_background_as_zeros"] = cfg["encode_background_as_zeros"]
        rpn_cfg["num_class"] = cfg["num_class"]
        rpn_cfg["use_direction_classifier"] = cfg["use_direction_classifier"]
        rpn_cfg["num_direction_bins"] = cfg["num_direction_bins"]
        encode_background_as_zeros = rpn_cfg["encode_background_as_zeros"]
        use_sigmoid_score = cfg["use_sigmoid_score"]
        self.voxel_feature_extractor = voxel_encoder.get_vfe_class(vfe_cfg["name"])(**vfe_cfg)
        middle_cfg_ = middle_cfg.copy()
        del middle_cfg_["downsample_factor"]
        self.middle_feature_extractor = middle.get_middle_class(middle_cfg["name"])(**middle_cfg_)
        self.rpn = rpn.get_rpn_class(rpn_cfg["name"])(**rpn_cfg)

        self.rpn_acc = metrics.Accuracy(
            dim=-1, encode_background_as_zeros=encode_background_as_zeros)
        self.rpn_precision = metrics.Precision(dim=-1)
        self.rpn_recall = metrics.Recall(dim=-1)
        self.rpn_metrics = metrics.PrecisionRecall(
            dim=-1,
            thresholds=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
            use_sigmoid_score=use_sigmoid_score,
            encode_background_as_zeros=encode_background_as_zeros)

        self.rpn_cls_loss = metrics.Scalar()
        self.rpn_loc_loss = metrics.Scalar()
        self.rpn_total_loss = metrics.Scalar()
        self.register_buffer("global_step", torch.LongTensor(1).zero_())

        self._time_dict = {}
        self._time_total_dict = {}
        self._time_count_dict = {}

    def network_forward(self, voxels, num_points, coors, batch_size):
        """this function is used for subclass.
        you can add custom network architecture by subclass VoxelNet class
        and override this function.
        Returns: 
            preds_dict: {
                box_preds: ...
                cls_preds: ...
                dir_cls_preds: ...
            }
        """
        voxel_features = self.voxel_feature_extractor(voxels, num_points,
                                                      coors)
        spatial_features = self.middle_feature_extractor(
            voxel_features, coors, batch_size)
        preds_dict = self.rpn(spatial_features)
        return preds_dict

    def forward(self, example):
        """module's forward should always accept dict and return loss.
        """
        voxels = example["voxels"]
        num_points = example["num_points"]
        coors = example["coordinates"]
        batch_anchors = example["anchors"]
        batch_size_dev = batch_anchors.shape[0]
        # features: [num_voxels, max_num_points_per_voxel, 7]
        # num_points: [num_voxels]
        # coors: [num_voxels, 4]
        preds_dict = self.network_forward(voxels, num_points, coors, batch_size_dev)
        # need to check size.
        box_preds = preds_dict["box_preds"].view(batch_size_dev, -1, self._box_coder.code_size)
        err_msg = f"num_anchors={batch_anchors.shape[1]}, but num_output={box_preds.shape[1]}. please check size"
        assert batch_anchors.shape[1] == box_preds.shape[1], err_msg
        if self.training:
            return self.loss(example, preds_dict)
        else:
            with torch.no_grad():
                res = self.predict(example, preds_dict)
            return res

    def predict(self, example, preds_dict):
        """start with v1.6.0, this function don't contain any kitti-specific code.
        Returns:
            predict: list of pred_dict.
            pred_dict: {
                box3d_lidar: [N, 7] 3d box.
                scores: [N]
                label_preds: [N]
                metadata: meta-data which contains dataset-specific information.
                    for kitti, it contains image idx (label idx), 
                    for nuscenes, sample_token is saved in it.
            }
        """
        batch_size = example['anchors'].shape[0]
        if "metadata" not in example or len(example["metadata"]) == 0:
            meta_list = [None] * batch_size
        else:
            meta_list = example["metadata"]
        batch_anchors = example["anchors"].view(batch_size, -1,
                                                example["anchors"].shape[-1])
        if "anchors_mask" not in example:
            batch_anchors_mask = [None] * batch_size
        else:
            batch_anchors_mask = example["anchors_mask"].view(batch_size, -1)

        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        batch_box_preds = batch_box_preds.view(batch_size, -1,
                                               self._box_coder.code_size)
        num_class_with_bg = self._num_class
        if not self._encode_background_as_zeros:
            num_class_with_bg = self._num_class + 1

        batch_cls_preds = batch_cls_preds.view(batch_size, -1,
                                               num_class_with_bg)
        batch_box_preds = self._box_coder.decode_torch(batch_box_preds,
                                                       batch_anchors)
        if self._use_direction_classifier:
            batch_dir_preds = preds_dict["dir_cls_preds"]
            batch_dir_preds = batch_dir_preds.view(batch_size, -1,
                                                   self._num_direction_bins)
        else:
            batch_dir_preds = [None] * batch_size

        predictions_dicts = []
        post_center_range = None
        if len(self._post_center_range) > 0:
            post_center_range = torch.tensor(
                self._post_center_range,
                dtype=batch_box_preds.dtype,
                device=batch_box_preds.device).float()
        for box_preds, cls_preds, dir_preds, a_mask, meta in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds,
                batch_anchors_mask, meta_list):
            if a_mask is not None:
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
            box_preds = box_preds.float()
            cls_preds = cls_preds.float()
            if self._use_direction_classifier:
                if a_mask is not None:
                    dir_preds = dir_preds[a_mask]
                dir_labels = torch.max(dir_preds, dim=-1)[1]
            if self._encode_background_as_zeros:
                # this don't support softmax
                assert self._use_sigmoid_score is True
                total_scores = torch.sigmoid(cls_preds)
            else:
                # encode background as first element in one-hot vector
                if self._use_sigmoid_score:
                    total_scores = torch.sigmoid(cls_preds)[..., 1:]
                else:
                    total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
            # Apply NMS in birdeye view
            if self._use_rotate_nms:
                nms_func = rotate_nms
            else:
                # nms_func = box_torch_ops.nms
                raise NotImplementedError
            feature_map_size_prod = batch_box_preds.shape[
                1] // self.target_assigner.num_anchors_per_location
            if self._multiclass_nms:
                raise NotImplementedError
            else:
                # get highest score per prediction, than apply nms
                # to remove overlapped box.
                if num_class_with_bg == 1:
                    top_scores = total_scores.squeeze(-1)
                    top_labels = torch.zeros(
                        total_scores.shape[0],
                        device=total_scores.device,
                        dtype=torch.long)
                else:
                    top_scores, top_labels = torch.max(
                        total_scores, dim=-1)
                if self._nms_score_thresholds[0] > 0.0:
                    top_scores_keep = top_scores >= self._nms_score_thresholds[0]
                    top_scores = top_scores.masked_select(top_scores_keep)

                if top_scores.shape[0] != 0:
                    if self._nms_score_thresholds[0] > 0.0:
                        box_preds = box_preds[top_scores_keep]
                        if self._use_direction_classifier:
                            dir_labels = dir_labels[top_scores_keep]
                        top_labels = top_labels[top_scores_keep]
                    boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                    if not self._use_rotate_nms:
                        box_preds_corners = center_to_corner_box2d(
                            boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                            boxes_for_nms[:, 4])
                        boxes_for_nms = corner_to_standup_nd(
                            box_preds_corners)
                    # the nms in 3d detection just remove overlap boxes.
                    selected = nms_func(
                        boxes_for_nms,
                        top_scores,
                        pre_max_size=self._nms_pre_max_sizes[0],
                        post_max_size=self._nms_post_max_sizes[0],
                        iou_threshold=self._nms_iou_thresholds[0],
                    )
                else:
                    selected = []
                # if selected is not None:
                selected_boxes = box_preds[selected]
                if self._use_direction_classifier:
                    selected_dir_labels = dir_labels[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]
            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                if self._use_direction_classifier:
                    dir_labels = selected_dir_labels
                    period = (2 * np.pi / self._num_direction_bins)
                    dir_rot = limit_period(
                        box_preds[..., 6] - self._dir_offset,
                        self._dir_limit_offset, period)
                    box_preds[
                        ...,
                        6] = dir_rot + self._dir_offset + period * dir_labels.to(
                            box_preds.dtype)
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_dict = {
                        "box3d_lidar": final_box_preds[mask],
                        "scores": final_scores[mask],
                        "label_preds": label_preds[mask],
                        "metadata": meta,
                    }
                else:
                    predictions_dict = {
                        "box3d_lidar": final_box_preds,
                        "scores": final_scores,
                        "label_preds": label_preds,
                        "metadata": meta,
                    }
            else:
                dtype = batch_box_preds.dtype
                device = batch_box_preds.device
                predictions_dict = {
                    "box3d_lidar":
                    torch.zeros([0, box_preds.shape[-1]],
                                dtype=dtype,
                                device=device),
                    "scores":
                    torch.zeros([0], dtype=dtype, device=device),
                    "label_preds":
                    torch.zeros([0], dtype=top_labels.dtype, device=device),
                    "metadata":
                    meta,
                }
            predictions_dicts.append(predictions_dict)
        return predictions_dicts

    def loss(self, example, preds_dict):
        box_preds = preds_dict["box_preds"]
        cls_preds = preds_dict["cls_preds"]
        batch_size_dev = cls_preds.shape[0]
        labels = example['labels']
        reg_targets = example['reg_targets']
        importance = example['importance']
        cls_weights, reg_weights, cared = prepare_loss_weights(
            labels,
            pos_cls_weight=self._pos_cls_weight,
            neg_cls_weight=self._neg_cls_weight,
            loss_norm_type=self._loss_norm_type,
            dtype=box_preds.dtype)

        cls_targets = labels * cared.type_as(labels)
        cls_targets = cls_targets.unsqueeze(-1)
        loc_loss, cls_loss = create_loss(
            self._loc_loss_ftor,
            self._cls_loss_ftor,
            box_preds=box_preds,
            cls_preds=cls_preds,
            cls_targets=cls_targets,
            cls_weights=cls_weights * importance,
            reg_targets=reg_targets,
            reg_weights=reg_weights * importance,
            num_class=self._num_class,
            encode_rad_error_by_sin=self._encode_rad_error_by_sin,
            encode_background_as_zeros=self._encode_background_as_zeros,
            box_code_size=self._box_coder.code_size,
            sin_error_factor=self._sin_error_factor,
            num_direction_bins=self._num_direction_bins,
        )
        loc_loss_reduced = loc_loss.sum() / batch_size_dev
        loc_loss_reduced *= self._loc_loss_weight
        cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)
        cls_pos_loss /= self._pos_cls_weight
        cls_neg_loss /= self._neg_cls_weight
        cls_loss_reduced = cls_loss.sum() / batch_size_dev
        cls_loss_reduced *= self._cls_loss_weight
        loss = loc_loss_reduced + cls_loss_reduced
        if self._use_direction_classifier:
            dir_targets = get_direction_target(
                example['anchors'],
                reg_targets,
                dir_offset=self._dir_offset,
                num_bins=self._num_direction_bins)
            dir_logits = preds_dict["dir_cls_preds"].view(
                batch_size_dev, -1, self._num_direction_bins)
            weights = (labels > 0).type_as(dir_logits) * importance
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self._dir_loss_ftor(
                dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size_dev
            loss += dir_loss * self._direction_loss_weight
        res = {
            "loss": loss,
            "cls_loss": cls_loss,
            "loc_loss": loc_loss,
            "cls_pos_loss": cls_pos_loss,
            "cls_neg_loss": cls_neg_loss,
            "cls_preds": cls_preds,
            "cls_loss_reduced": cls_loss_reduced,
            "loc_loss_reduced": loc_loss_reduced,
            "cared": cared,
        }
        if self._use_direction_classifier:
            res["dir_loss_reduced"] = dir_loss
        return res

def prepare_loss_weights(labels,
                         pos_cls_weight=1.0,
                         neg_cls_weight=1.0,
                         loss_norm_type="NormByNumPositives",
                         dtype=torch.float32):
    """get cls_weights and reg_weights from labels.
    """
    cared = labels >= 0
    # cared: [N, num_anchors]
    positives = labels > 0
    negatives = labels == 0
    negative_cls_weights = negatives.type(dtype) * neg_cls_weight
    cls_weights = negative_cls_weights + pos_cls_weight * positives.type(dtype)
    reg_weights = positives.type(dtype)
    if loss_norm_type == "NormByNumPositives":  # for focal loss
        pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
    else:
        raise NotImplementedError
    return cls_weights, reg_weights, cared

def create_loss(loc_loss_ftor,
                cls_loss_ftor,
                box_preds,
                cls_preds,
                cls_targets,
                cls_weights,
                reg_targets,
                reg_weights,
                num_class,
                encode_background_as_zeros=True,
                encode_rad_error_by_sin=True,
                sin_error_factor=1.0,
                box_code_size=7,
                num_direction_bins=2):
    batch_size = int(box_preds.shape[0])
    box_preds = box_preds.view(batch_size, -1, box_code_size)
    if encode_background_as_zeros:
        cls_preds = cls_preds.view(batch_size, -1, num_class)
    else:
        cls_preds = cls_preds.view(batch_size, -1, num_class + 1)
    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = one_hot(
        cls_targets, depth=num_class + 1, dtype=box_preds.dtype)
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]
    if encode_rad_error_by_sin:
        # sin(a - b) = sinacosb-cosasinb
        # reg_tg_rot = box_torch_ops.limit_period(
        #     reg_targets[..., 6:7], 0.5, 2 * np.pi / num_direction_bins)
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets, box_preds[..., 6:7], reg_targets[..., 6:7],
                                                    sin_error_factor)

    loc_losses = loc_loss_ftor(
        box_preds, reg_targets, weights=reg_weights)  # [N, M]
    cls_losses = cls_loss_ftor(
        cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
    return loc_losses, cls_losses

def add_sin_difference(boxes1, boxes2, boxes1_rot, boxes2_rot, factor=1.0):
    if factor != 1.0:
        boxes1_rot = factor * boxes1_rot
        boxes2_rot = factor * boxes2_rot
    rad_pred_encoding = torch.sin(boxes1_rot) * torch.cos(boxes2_rot)
    rad_tg_encoding = torch.cos(boxes1_rot) * torch.sin(boxes2_rot)
    boxes1 = torch.cat([boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]],
                       dim=-1)
    boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]],
                       dim=-1)
    return boxes1, boxes2

def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss

def get_direction_target(anchors,
                         reg_targets,
                         one_hot=True,
                         dir_offset=0,
                         num_bins=2):
    batch_size = reg_targets.shape[0]
    anchors = anchors.view(batch_size, -1, anchors.shape[-1])
    rot_gt = reg_targets[..., 6] + anchors[..., 6]
    offset_rot = limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
    dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
    dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
    if one_hot:
        dir_cls_targets = one_hot(
            dir_cls_targets, num_bins, dtype=anchors.dtype)
    return dir_cls_targets

if __name__ == "__main__":
    from det3.methods.second.configs.config import cfg
    from det3.methods.second.builder import (voxelizer_builder, box_coder_builder,
                                            similarity_calculator_builder, 
                                            anchor_generator_builder, target_assigner_builder)
    voxelizer = voxelizer_builder.build(voxelizer_cfg=cfg.Voxelizer)
    anchor_generator = anchor_generator_builder.build(anchor_generator_cfg=cfg.AnchorGenerator)
    box_coder = box_coder_builder.build(box_coder_cfg=cfg.BoxCoder)
    similarity_calculator = similarity_calculator_builder.build(similarity_calculator_cfg=cfg.SimilarityCalculator)
    target_assigner = target_assigner_builder.build(target_assigner_cfg=cfg.TargetAssigner,
                                                    box_coder=box_coder,
                                                    anchor_generators=[anchor_generator],
                                                    region_similarity_calculators=[similarity_calculator])
    grid_size = voxelizer.grid_size
    cfg.Net["MiddleLayer"]["output_shape"] = [1] + grid_size[::-1].tolist() + [16] # Q: Why 16 here? 
    cfg.Net["RPN"]["num_anchor_per_loc"] = target_assigner.num_anchors_per_location
    cfg.Net["RPN"]["box_code_size"] = target_assigner.box_coder.code_size
    cfg.Net["num_input_features"] = 4
    voxelnet = VoxelNet(vfe_cfg=cfg.Net["VoxelEncoder"],
                        middle_cfg=cfg.Net["MiddleLayer"],
                        rpn_cfg=cfg.Net["RPN"],
                        cls_loss_cfg=cfg.Net["ClassificationLoss"],
                        loc_loss_cfg=cfg.Net["LocalizationLoss"],
                        cfg=cfg.Net,
                        target_assigner=target_assigner,
                        voxelizer=voxelizer)