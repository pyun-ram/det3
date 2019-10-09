class BaseBoxCoder:
    def __init__(self):
        raise NotImplementedError
    
    def encode(label):
        raise NotImplementedError
    
    def decode(grid):
        raise NotImplementedError
    
class BoxCoderV1(BaseBoxCoder):
    def __init__(self):
        pass

    def encode(boxes, anchors)