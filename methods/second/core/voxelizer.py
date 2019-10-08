class BaseVoxelizer:
    def __init__(self):
        raise NotImplementedError
    
    def apply(self, pc):
        raise NotImplementedError