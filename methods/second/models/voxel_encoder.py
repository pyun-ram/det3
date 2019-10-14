import torch.nn as nn

REGISTERED_VFE_CLASSES = {}
def register_vfe(cls, name=None):
    '''Source: https://github.com/traveller59/second.pytorch'''
    global REGISTERED_VFE_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_VFE_CLASSES, f"exist class: {REGISTERED_VFE_CLASSES}"
    REGISTERED_VFE_CLASSES[name] = cls
    return cls
def get_vfe_class(name):
    '''Source: https://github.com/traveller59/second.pytorch'''
    global REGISTERED_VFE_CLASSES
    assert name in REGISTERED_VFE_CLASSES, f"available class: {REGISTERED_VFE_CLASSES}"
    return REGISTERED_VFE_CLASSES[name]

@register_vfe
class SimpleVoxel(nn.Module):
    '''
    Source: https://github.com/traveller59/second.pytorch
    '''
    def __init__(self,
                 num_input_features=4,
                 name='SimpleVoxel'):

        super(SimpleVoxel, self).__init__()
        self.name = name
        self.num_input_features = num_input_features

    def forward(self, features, num_voxels, coors):
        points_mean = features[:, :, :self.num_input_features].sum(
            dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        return points_mean.contiguous()