from det3.methods.second.utils.import_tool import load_module

def build(anchor_generator_cfg):
    class_name = anchor_generator_cfg["type"]
    params = {k:v for k, v in anchor_generator_cfg.items() if k != "type"}
    builder = load_module("methods/second/core/anchor_generator.py", name=class_name)
    return builder(**params)

