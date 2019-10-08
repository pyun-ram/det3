class BaseNet:
    def __init__(self):
        raise NotImplementedError
    
    def forward(self, input:dict)-> dict:
        raise NotImplementedError