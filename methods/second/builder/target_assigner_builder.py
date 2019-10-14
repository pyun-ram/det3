from det3.methods.second.utils.import_tool import load_module

def build(target_assigner_cfg,
          box_coder,
          anchor_generators,
          region_similarity_calculators):
    class_name = target_assigner_cfg["type"]
    params = {k:v for k, v in target_assigner_cfg.items() if k != "type"}
    params["box_coder"] = box_coder
    params["anchor_generators"] = anchor_generators
    params["region_similarity_calculators"] = region_similarity_calculators
    builder = load_module("methods/second/core/target_assigner.py", name=class_name)
    return builder(**params)