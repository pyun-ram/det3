'''
File Created: Friday, 22nd March 2019 10:28:47 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import numpy as np
from PIL import Image, ImageDraw
try:
    from ..dataloader.kittidata import KittiObj
    from ..dataloader.carladata import CarlaObj, CarlaCalib
    from ..utils.utils import istype
except:
    # Run script
    from det3.dataloader.kittidata import KittiObj
    from det3.dataloader.carladata import CarlaObj, CarlaCalib
    from det3.utils.utils import istype

class BEVImage:
    '''
    class of Bird's Eye View Image
    The following data is:
        KittiData: LiDAR Frame
        CarlaData: Imu Frame
    '''
    def __init__(self, x_range, y_range, grid_size):
        '''
        initialization (The arguments are all in (LiDAR/IMU Frame).
        x_range(tuple): (min_x(float), max_x(float))
        y_range(tuple): (min_y(float), max_y(float))
        grid_size(tuple): (dx, dy)
        '''
        self.x_range = x_range
        self.y_range = y_range
        self.grid_size = grid_size
        self.data = None

    def from_lidar(self, pc, scale=1):
        '''
        convert point cloud into a BEV Image
        inputs:
            ps(np.array): point cloud with shape [# of points, >=3]
                in LiDAR/IMU Frame
            scale(int): size of the points in BEV Image
        implicitly return:
            self.data(np.array): [height, width, 3]
                white background with blue points
        '''
        min_x, max_x = self.x_range
        min_y, max_y = self.y_range
        dx, dy = self.grid_size
        height = np.floor((max_y - min_y) / dy).astype(np.int)
        width = np.floor((max_x - min_x) / dx).astype(np.int)
        bevimg = np.zeros((height, width))
        pc_BEV = self.lidar2BEV(pc[:, :3])
        num_of_pts = 0
        for (x, y) in pc_BEV:
            if scale < x < width-scale and scale < y < height-scale:
                bevimg[y-scale:y+scale, x-scale:x+scale] += 1
                num_of_pts += 1
        bevimg = bevimg - np.min(bevimg)
        divisor = np.max(bevimg) - np.min(bevimg)
        factor = 100 * num_of_pts / 4000 # for better visualization
        bevimg = np.clip((bevimg / divisor * 255.0 * 100), a_min=0, a_max=255)
        # if blue pts and white background
        # bevimg = (255 - bevimg).astype(np.uint8)
        # tmp = np.ones((height, width, 3)).astype(np.uint8) * 255
        # tmp[:, :, 0] = bevimg
        # tmp[:, :, 1] = bevimg
        # self.data = tmp
        # white pts and black back ground
        self.data = np.tile(bevimg.reshape(height, width, 1), 3).astype(np.uint8)
        return self

    def lidar2BEV(self, pts):
        '''
        transform the pts from FLiDAR/FCARLA to BEV coordinate
        inputs:
            pts (np.array): [#pts, 3] in FLiDAR/FCARLA
        return:
            pts_BEV (np.array): [#pts, 2] in np.int
                points in BEV Frame [row, col]
            Note: There are some points might out of the BEV coordinate
                (i.e. not in range [height, width])
        '''
        min_x, _ = self.x_range
        min_y, max_y = self.y_range
        dx, dy = self.grid_size
        height = np.floor((max_y - min_y) / dy).astype(np.int)
        x, y = np.floor((pts[:, :1] -min_x) / dx).astype(np.int), height - np.floor((pts[:, 1:2] -min_y) / dy).astype(np.int)
        pts_BEV = np.hstack([x, y])
        return pts_BEV

    def draw_box(self, obj, calib, bool_gt=False, width=3, c=None, text=None):
        '''
        draw bounding box on BEV Image
        inputs:
            obj (KittiObj/CarlaObj)
            calib (KittiCalib/CarlaCalib)
            Note: It is able to hundle the out-of-coordinate bounding boxes.
                gt: purple
                est: yellow
        '''
        if self.data is None:
            print("from_lidar should be run first")
            raise RuntimeError
        if istype(obj, 'KittiObj') and istype(calib, 'KittiCalib'):
            cns_Fcam = obj.get_bbox3dcorners()[:4, :]
            cns_Flidar = calib.leftcam2lidar(cns_Fcam)
            cns_FBEV = self.lidar2BEV(cns_Flidar)
        elif istype(obj, 'CarlaObj') and istype(calib, 'CarlaCalib'):
            cns_Fimu = obj.get_bbox3dcorners()[:4, :]
            cns_FBEV = self.lidar2BEV(cns_Fimu)
        elif istype(obj, 'WaymoObj') and istype(calib, 'WaymoCalib'):
            cns_Fimu = obj.get_bbox3dcorners()[:4, :]
            cns_FBEV = self.lidar2BEV(cns_Fimu)
        else:
            print(obj.__class__)
            print(calib.__class__)
            raise NotImplementedError

        bev_img = Image.fromarray(self.data)
        draw = ImageDraw.Draw(bev_img)
        p1, p2, p3, p4 = cns_FBEV
        color = 'purple' if bool_gt else 'yellow'
        color = c if c is not None else color
        draw.line([p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1], p1[0], p1[1]], fill=color, width=width)
        if text is not None:
            draw.text((p1[0], p1[1]), text)
        self.data = np.array(bev_img)
        return self

    def save(self, path:str):
        assert self.data is not None
        from PIL import Image
        tmp = Image.fromarray(self.data)
        tmp.save(path)

class FVImage:
    '''
    class of Front View Image
    '''
    def __init__(self):
        '''
        initilization
        '''
        self.data = None
    def from_image(self, image):
        '''
        load image.
        inputs:
            image (np.array): [h, w, 3]
        '''
        self.data = image
        return self
    def from_lidar(self, calib, pts):
        '''
        project pts from IMU frame to camera plane specified by calib
        inputs:
            calib (CarlaCalib)
            pts (np.array) [#pts, >=3]
                pts in FIMU (CARLA) or in FLidar (Kitti)
        '''
        if istype(calib, "CarlaCalib"):
            pts_Fcam = calib.imu2cam(pts)
            pts_Fimg = calib.cam2imgplane(pts_Fcam)
            width = np.ceil(calib.P0[0, 2] * 2).astype(np.int)
            height = np.ceil(calib.P0[1, 2] * 2).astype(np.int)
        elif istype(calib, "KittiCalib"):
            pts_Fcam = calib.lidar2leftcam(pts[:, :3])
            pts_Fimg = calib.leftcam2imgplane(pts_Fcam)
            width = np.ceil(calib.P2[0, 2] * 2).astype(np.int)
            height = np.ceil(calib.P2[1, 2] * 2).astype(np.int)
        elif istype(calib, "WaymoCalib"):
            pts_Fcam = calib.imu2cam(pts)
            pts_Fimg = calib.cam2imgplane(pts_Fcam)
            width = np.ceil(calib.P0[0, 2] * 2).astype(np.int)
            height = np.ceil(calib.P0[1, 2] * 2).astype(np.int)
        else:
            raise NotImplementedError

        self.data = np.zeros((height, width))
        for (x, y), (x_, y_, z_) in zip(pts_Fimg, pts_Fcam):
            x, y = int(x), int(y)
            if 0 <= x < width and 0 <= y < height:
                self.data[y, x] = (np.sqrt(x_*x_ + y_*y_ + z_*z_) if self.data[y, x] == 0
                                   else min(np.sqrt(x_*x_ + y_*y_ + z_*z_), self.data[y, x]))
        self.data = self.data - np.min(self.data)
        divisor = np.max(self.data) - np.min(self.data)
        self.data = np.clip(self.data / divisor * 255 * 2, a_min=0, a_max=255)
        self.data = np.tile(self.data.reshape(height, width, 1), 3).astype(np.uint8)
        return self

    def draw_box(self, obj, calib, bool_gt=False, width=3, c=None):
        '''
        draw bounding box on Front View Image
        inputs:
            obj (KittiObj/CalibObj)
            calib (KittiCalib/CalibObj)
            Note: It is able to hundle the out-of-coordinate bounding boxes.
                gt: purple
                est: yellow
            Note: The 2D bounding box is computed from 3D bounding box
        '''
        if self.data is None:
            print("from_lidar should be run first")
            raise RuntimeError
        if istype(obj, "KittiObj") and istype(calib, "KittiCalib"):
            cns_Fcam = obj.get_bbox3dcorners()
            cns_Fcam2d = calib.leftcam2imgplane(cns_Fcam)
        elif istype(obj, "CarlaObj") and istype(calib, "CarlaCalib"):
            cns_Fimu = obj.get_bbox3dcorners()
            cns_Fcam = calib.imu2cam(cns_Fimu)
            cns_Fcam2d = calib.cam2imgplane(cns_Fcam)
        elif istype(obj, "WaymoObj") and istype(calib, "WaymoCalib"):
            cns_Fimu = obj.get_bbox3dcorners()
            cns_Fcam = calib.imu2cam(cns_Fimu)
            cns_Fcam2d = calib.cam2imgplane(cns_Fcam)
        else:
            raise NotImplementedError
        minx = int(np.min(cns_Fcam2d[:, 0]))
        maxx = int(np.max(cns_Fcam2d[:, 0]))
        miny = int(np.min(cns_Fcam2d[:, 1]))
        maxy = int(np.max(cns_Fcam2d[:, 1]))
        fv_img = Image.fromarray(self.data)
        draw = ImageDraw.Draw(fv_img)
        p1 = np.array([minx, miny])
        p2 = np.array([minx, maxy])
        p3 = np.array([maxx, maxy])
        p4 = np.array([maxx, miny])
        color = 'purple' if bool_gt else 'yellow'
        color = c if c is not None else color
        draw.line([p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1], p1[0], p1[1]],
                  fill=color, width=width)
        self.data = np.array(fv_img)
        return self

    def draw_3dbox(self, obj, calib, bool_gt=False, width=3, c=None):
        '''
        draw bounding box on Front View Image
        inputs:
            obj (KittiObj/CalibObj)
            calib (KittiCalib/CalibObj)
            Note: It is able to hundle the out-of-coordinate bounding boxes.
                gt: purple
                est: yellow
            Note: The 2D bounding box is computed from 3D bounding box
        '''
        if self.data is None:
            print("from_lidar should be run first")
            raise RuntimeError
        if istype(obj, "KittiObj") and istype(calib, "KittiCalib"):
            cns_Fcam = obj.get_bbox3dcorners()
            cns_Fcam2d = calib.leftcam2imgplane(cns_Fcam)
        elif istype(obj, "CarlaObj") and istype(calib, "CarlaCalib"):
            cns_Fimu = obj.get_bbox3dcorners()
            cns_Fcam = calib.imu2cam(cns_Fimu)
            cns_Fcam2d = calib.cam2imgplane(cns_Fcam)
        elif istype(obj, "WaymoObj") and istype(calib, "WaymoCalib"):
            cns_Fimu = obj.get_bbox3dcorners()
            cns_Fcam = calib.imu2cam(cns_Fimu)
            cns_Fcam2d = calib.cam2imgplane(cns_Fcam)
        else:
            raise NotImplementedError
        color = 'purple' if bool_gt else 'yellow'
        color = c if c is not None else color
        fv_img = Image.fromarray(self.data)
        draw = ImageDraw.Draw(fv_img)
        draw.line([cns_Fcam2d[0, 0], cns_Fcam2d[0, 1],
                   cns_Fcam2d[1, 0], cns_Fcam2d[1, 1],
                   cns_Fcam2d[2, 0], cns_Fcam2d[2, 1],
                   cns_Fcam2d[3, 0], cns_Fcam2d[3, 1],
                   cns_Fcam2d[0, 0], cns_Fcam2d[0, 1]], fill=color, width=width)
        draw.line([cns_Fcam2d[4, 0], cns_Fcam2d[4, 1],
                   cns_Fcam2d[5, 0], cns_Fcam2d[5, 1],
                   cns_Fcam2d[6, 0], cns_Fcam2d[6, 1],
                   cns_Fcam2d[7, 0], cns_Fcam2d[7, 1],
                   cns_Fcam2d[4, 0], cns_Fcam2d[4, 1]], fill=color, width=width)
        draw.line([cns_Fcam2d[0, 0], cns_Fcam2d[0, 1],
                   cns_Fcam2d[1, 0], cns_Fcam2d[1, 1],
                   cns_Fcam2d[5, 0], cns_Fcam2d[5, 1],
                   cns_Fcam2d[4, 0], cns_Fcam2d[4, 1],
                   cns_Fcam2d[0, 0], cns_Fcam2d[0, 1]], fill=color, width=width)
        draw.line([cns_Fcam2d[2, 0], cns_Fcam2d[2, 1],
                   cns_Fcam2d[3, 0], cns_Fcam2d[3, 1],
                   cns_Fcam2d[7, 0], cns_Fcam2d[7, 1],
                   cns_Fcam2d[6, 0], cns_Fcam2d[6, 1],
                   cns_Fcam2d[2, 0], cns_Fcam2d[2, 1]], fill=color, width=width)
        self.data = np.array(fv_img)
        return self

    def save(self, path:str):
        assert self.data is not None
        from PIL import Image
        tmp = Image.fromarray(self.data)
        tmp.save(path)

if __name__ == "__main__":
    from det3.dataloader.kittidata import KittiData
    from det3.dataloader.carladata import CarlaData
    from PIL import Image
    for i in range(100, 300):
        tag = "{:06d}".format(i)
        pc, label, calib = CarlaData('/usr/app/data/CARLA/dev', tag).read_data()
        bevimg = BEVImage(x_range=(-100, 100), y_range=(-50,50), grid_size=(0.05, 0.05))
        # bevimg.from_lidar(np.vstack([pc['velo_top'], pc['velo_left'], pc['velo_right']]), scale=1)
        bevimg.from_lidar(np.vstack([pc['velo_top']]), scale=1)
        for obj in label.read_label_file().data:
            if obj.type == 'Car':
                bevimg.draw_box(obj, calib, bool_gt=False)
        bevimg_img = Image.fromarray(bevimg.data)
        bevimg_img.save("/usr/app/vis/dev/{}.png".format(tag))

    # data = KittiData('/usr/app/data/KITTI/dev/', '000000')
    # calib, _, label, pc = data.read_data()
    # bevimg = BEVImage(x_range=(0, 70), y_range=(-30,30), grid_size=(0.05, 0.05))
    # bevimg.from_lidar(pc, scale=1)
    # for obj in label.read_label_file().data:
    #     if obj.type == 'Pedestrian':
    #         bevimg.draw_box(obj, calib, bool_gt=False)
    #     print(obj)
    # bevimg_img = Image.fromarray(bevimg.data)
    # bevimg_img.save("lala.png")

    # data = KittiData('/usr/app/data/KITTI/dev/', '000007')
    # calib, img, label, pc = data.read_data()
    # fvimg = FVImage()
    # fvimg.from_image(img)
    # for obj in label.read_label_file().data:
    #     if obj.type == 'Car':
    #         fvimg.draw_box(obj, calib, bool_gt=False)
    #     print(obj)
    # fvimg_img = Image.fromarray(fvimg.data)
    # fvimg_img.save("/usr/app/vis/dev/fv.png")    

