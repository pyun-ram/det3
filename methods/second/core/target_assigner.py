import numpy as np
from det3.methods.second.utils.log_tool import Logger
from det3.methods.second.ops.ops import create_target_np

class BaseTaskAssigner:
    def __init__(self, box_coder, anchor_generators):
        raise NotImplementedError
    
    @property
    def box_coder(self):
        raise NotImplementedError

    @property
    def anchor_generators(self):
        raise NotImplementedError

    def apply(self, anchors, gt_boxes):
        raise NotImplementedError

class TaskAssignerV1(BaseTaskAssigner):
    '''
    Source: https://github.com/traveller59/second.pytorch
    '''
    def __init__(self,
                 box_coder,
                 anchor_generators,
                 classes,
                 feature_map_sizes,
                 positive_fraction=None,
                 region_similarity_calculators=None,
                 sample_size=512,
                 assign_per_class=True):
        self._box_coder = box_coder
        self._anchor_generators = anchor_generators
        self._sim_calcs = region_similarity_calculators
        box_ndims = [a.ndim for a in anchor_generators]
        assert all([e == box_ndims[0] for e in box_ndims])
        self._positive_fraction = positive_fraction
        self._sample_size = sample_size
        self._classes = classes
        self._assign_per_class = assign_per_class
        self._feature_map_sizes = feature_map_sizes
        Logger.log_txt("Warning: TaskAssignerV1 requires unit-test.")

    @property
    def box_coder(self):
        return self._box_coder

    @property
    def anchor_generators(self):
        return self._anchor_generators

    @property
    def classes(self):
        return self._classes

    def assign(self,
               anchors,
               anchors_dict,
               gt_boxes,
               anchors_mask=None,
               gt_classes=None,
               gt_names=None,
               matched_thresholds=None,
               unmatched_thresholds=None,
               importance=None):
        if self._assign_per_class:
            return self.assign_per_class(anchors_dict, gt_boxes, anchors_mask,
                                         gt_classes, gt_names, importance=importance)
        else:
            raise NotImplementedError

    def assign_per_class(self,
                         anchors_dict,
                         gt_boxes,
                         anchors_mask=None,
                         gt_classes=None,
                         gt_names=None,
                         importance=None):
        """this function assign target individally for each class.
        recommend for multi-class network.
        """
        def box_encoding_fn(boxes, anchors):
            return self._box_coder.encode(boxes, anchors)

        targets_list = []
        anchor_loc_idx = 0
        anchor_gene_idx = 0
        for class_name, anchor_dict in anchors_dict.items():
            def similarity_fn(anchors, gt_boxes):
                anchors_rbv = anchors[:, [0, 1, 3, 4, 6]]
                gt_boxes_rbv = gt_boxes[:, [0, 1, 3, 4, 6]]
                return self._sim_calcs[anchor_gene_idx].compare(
                    anchors_rbv, gt_boxes_rbv)
            mask = np.array([c == class_name for c in gt_names], dtype=np.bool_)
            feature_map_size = anchor_dict["anchors"].shape[:3]
            num_loc = anchor_dict["anchors"].shape[-2]
            if anchors_mask is not None:
                anchors_mask = anchors_mask.reshape(-1)
                a_range = self.anchors_range(class_name)
                anchors_mask_class = anchors_mask[a_range[0]:a_range[1]].reshape(-1)
                prune_anchor_fn = lambda _: np.where(anchors_mask_class)[0]
            else:
                prune_anchor_fn = None
            targets = create_target_np(
                anchor_dict["anchors"].reshape(-1, self.box_ndim),
                gt_boxes[mask],
                similarity_fn,
                box_encoding_fn,
                prune_anchor_fn=prune_anchor_fn,
                gt_classes=gt_classes[mask],
                matched_threshold=anchor_dict["matched_thresholds"],
                unmatched_threshold=anchor_dict["unmatched_thresholds"],
                positive_fraction=self._positive_fraction,
                rpn_batch_size=self._sample_size,
                norm_by_num_examples=False,
                box_code_size=self.box_coder.code_size,
                gt_importance=importance)
            anchor_loc_idx += num_loc
            targets_list.append(targets)
            anchor_gene_idx += 1

        targets_dict = {
            "labels": [t["labels"] for t in targets_list],
            "bbox_targets": [t["bbox_targets"] for t in targets_list],
            "importance": [t["importance"] for t in targets_list],
        }
        targets_dict["bbox_targets"] = np.concatenate([
            v.reshape(-1, self.box_coder.code_size)
            for v in targets_dict["bbox_targets"]
        ],
                                                      axis=0)
        targets_dict["bbox_targets"] = targets_dict["bbox_targets"].reshape(
            -1, self.box_coder.code_size)
        targets_dict["labels"] = np.concatenate(
            [v.reshape(-1) for v in targets_dict["labels"]],
            axis=0)
        targets_dict["importance"] = np.concatenate(
            [v.reshape(-1) for v in targets_dict["importance"]],
            axis=0)
        targets_dict["labels"] = targets_dict["labels"].reshape(-1)
        targets_dict["importance"] = targets_dict["importance"].reshape(-1)

        return targets_dict

    def anchors_range(self, class_name):
        if isinstance(class_name, int):
            class_name = self._classes[class_name]
        assert class_name in self._classes
        num_anchors = 0
        anchor_ranges = []
        for name in self._classes:
            anchor_ranges.append((num_anchors, num_anchors + self.num_anchors(name)))
            num_anchors += anchor_ranges[-1][1] - num_anchors
        return anchor_ranges[self._classes.index(class_name)]

    @property
    def box_ndim(self):
        return self._anchor_generators[0].ndim

    def num_anchors(self, class_name):
        if isinstance(class_name, int):
            class_name = self._classes[class_name]
        assert class_name in self._classes
        class_idx = self._classes.index(class_name)
        ag = self._anchor_generators[class_idx]
        feature_map_size = self._feature_map_sizes[class_idx]
        return np.prod(feature_map_size) * ag.num_anchors_per_localization

    @property
    def num_anchors_per_location(self):
        num = 0
        for a_generator in self._anchor_generators:
            num += a_generator.num_anchors_per_localization
        return num