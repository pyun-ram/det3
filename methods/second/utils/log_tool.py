import os
class Logger:
    """For simple log.
    generate 3 kinds of log: 
    1. simple log.txt, all metric dicts are flattened to produce
    readable results.
    2. TensorBoard scalars and images
    3. save images for visualization
    once it is initilized, all later usage will based on the initilized path
    """
    global_dir = None
    def __init__(self):
        pass

    @property
    def global_dir(self):
        return Logger.global_dir

    @global_dir.setter
    def global_dir(self, v):
        assert os.path.isdir(v), f"{v} is not a valid dir."
        Logger.global_dir=v
    
    @staticmethod
    def log_txt(self, s):
        raise NotImplementedError

    @staticmethod
    def log_img(self, img, path):
        raise NotImplementedError
    
    @staticmethod
    def log_tsbd_scalor(self, img, v):
        raise NotImplementedError

    @staticmethod
    def log_tsbd_img(self, img):
        raise NotImplementedError

if __name__ == "__main__":
    logger1 = Logger()
    logger1.global_dir = "/usr/app"
    logger2 = Logger()
    print(logger1.global_dir)
    print(logger2.global_dir)