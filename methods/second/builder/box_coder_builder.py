from det3.methods.second.utils.import_tool import load_module

def build(box_coder_cfg):
    class_name = box_coder_cfg["type"]
    params = {k:v for k, v in box_coder_cfg.items() if k != "type"}
    builder = load_module("methods/second/core/box_coder.py", name=class_name)
    return builder(**params)