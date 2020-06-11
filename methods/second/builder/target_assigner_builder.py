from det3.methods.second.utils.import_tool import load_module
from det3.methods.second.builder.anchor_generator_builder import build as build_anchor_generator
from det3.methods.second.builder.similarity_calculator_builder import build as build_similarity_calculator

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

def build_multiclass(target_assigner_cfg,
      box_coder):
      class_name = target_assigner_cfg["type"]
      params = {k:v for k, v in target_assigner_cfg.items() if k != "type" and "class_settings" not in k}
      params["box_coder"] = box_coder
      # build anchors
      # build region similarity calculators
      classsettings_cfgs = [v for k, v in target_assigner_cfg.items() if "class_settings" in k]
      anchor_generators = []
      similarity_calculators = []
      classes = params["classes"]
      for cls in classes:
            classsetting_cfg = [itm for itm in classsettings_cfgs
                  if itm["AnchorGenerator"]["class_name"] == cls][0]
            anchor_generator_cfg = classsetting_cfg["AnchorGenerator"]
            anchor_generator = build_anchor_generator(anchor_generator_cfg)
            anchor_generators.append(anchor_generator)
            similarity_calculator_cfg = classsetting_cfg["SimilarityCalculator"]
            similarity_calculator = build_similarity_calculator(similarity_calculator_cfg)
            similarity_calculators.append(similarity_calculator)
      params["anchor_generators"] = anchor_generators
      params["region_similarity_calculators"] = similarity_calculators
      builder = load_module("methods/second/core/target_assigner.py", name=class_name)
      return builder(**params)
