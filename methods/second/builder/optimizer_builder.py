from functools import partial
from det3.methods.second.core.optimizer import OptimWrapper
from det3.methods.second.utils.torch_utils import OneCycle
import torch
import torch.nn as nn

def children(m: nn.Module):
    "Get children of `m`."
    return list(m.children())


def num_children(m: nn.Module) -> int:
    "Get number of children modules in `m`."
    return len(children(m))

flatten_model = lambda m: sum(map(flatten_model,m.children()),[]) if num_children(m) else [m]
get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

def build(optimizer_cfg, lr_scheduler_cfg, net):
    if optimizer_cfg["name"] == "ADAMOptimizer":
        optimizer_func = partial(
            torch.optim.Adam, betas=(0.9, 0.99), amsgrad=optimizer_cfg["amsgrad"])
        optimizer = OptimWrapper.create(
        optimizer_func,
        3e-3,
        get_layer_groups(net),
        wd=optimizer_cfg["weight_decay"],
        true_wd=optimizer_cfg["fixed_weight_decay"],
        bn_wd=True)
    else:
        raise NotImplementedError
    total_step = optimizer_cfg["steps"]
    if lr_scheduler_cfg["name"] == "OneCycle":
        lr_scheduler = OneCycle(optimizer, total_step, 
                                    lr_scheduler_cfg["lr_max"],
                                    list(lr_scheduler_cfg["moms"]),
                                    lr_scheduler_cfg["div_factor"],
                                    lr_scheduler_cfg["pct_start"])
    return optimizer, lr_scheduler