class BaseDataloader:
    def __init__(self):
        raise NotImplementedError
    
    def __get_item__(self, idx):
        raise NotImplementedError