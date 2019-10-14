from det3.methods.second.utils.import_tool import load_module

def build(cfg, target_assigner, voxelizer):
    class_name = cfg["name"]
    cfg["MiddleLayer"]["output_shape"] = [1] + voxelizer.grid_size[::-1].tolist() + [16] # Q: Why 16 here? 
    cfg["RPN"]["num_anchor_per_loc"] = target_assigner.num_anchors_per_location
    cfg["RPN"]["box_code_size"] = target_assigner.box_coder.code_size
    cfg["num_input_features"] = 4
    builder = load_module("methods/second/core/second.py", name=class_name)
    second = builder(vfe_cfg=cfg["VoxelEncoder"],
                     middle_cfg=cfg["MiddleLayer"],
                     rpn_cfg=cfg["RPN"],
                     cls_loss_cfg=cfg["ClassificationLoss"],
                     loc_loss_cfg=cfg["LocalizationLoss"],
                     cfg=cfg,
                     target_assigner=target_assigner,
                     voxelizer=voxelizer)
    return second