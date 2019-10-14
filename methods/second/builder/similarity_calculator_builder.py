from det3.methods.second.utils.import_tool import load_module

def build(similarity_calculator_cfg):
    class_name = similarity_calculator_cfg["type"]
    params = {k:v for k, v in similarity_calculator_cfg.items() if k != "type"}
    builder = load_module("methods/second/core/similarity_calculator.py", name=class_name)
    return builder(**params)