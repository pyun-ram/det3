'''
File Created: Friday, 22nd March 2019 10:28:47 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
'''
import numpy as np
from PIL import Image, ImageDraw

class BEVImage:
    '''
    class of Bird's Eye View Image
    '''
    def __init__(self, x_range, y_range, grid_size):
        '''
        initialization (The arguments are all in LiDAR Frame.
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
        for (x, y) in pc_BEV:
            if scale < x < width-scale and scale < y < height-scale:
                bevimg[y-scale:y+scale, x-scale:x+scale] += 1
        bevimg = bevimg - np.min(bevimg)
        divisor = np.max(bevimg) - np.min(bevimg)
        bevimg = np.clip((bevimg / divisor * 255.0 * 5), a_min=0, a_max=255) # the '*5' here is a hard code for better visualization
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
        transform the pts from the lidar frame to BEV coordinate
        inputs:
            pts (np.array): [#pts, 3]
                points in Lidar Frame [x, y, z]
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

    def draw_box(self, obj, calib, bool_gt=False, width=3):
        '''
        draw bounding box on BEV Image
        inputs:
            obj (KittiObj)
            calib (KittiCalib)
            Note: It is able to hundle the out-of-coordinate bounding boxes.
                gt: purple
                est: yellow
        '''
        if self.data is None:
            print("from_lidar should be run first")
            raise RuntimeError
        cns_Fcam = obj.get_bbox3dcorners()[:4, :]
        cns_Flidar = calib.leftcam2lidar(cns_Fcam)
        cns_FBEV = self.lidar2BEV(cns_Flidar)
        bev_img = Image.fromarray(self.data)
        draw = ImageDraw.Draw(bev_img)
        p1, p2, p3, p4 = cns_FBEV
        color = 'purple' if bool_gt else 'yellow'
        draw.line([p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1], p1[0], p1[1]], fill=color, width=width)
        self.data = np.array(bev_img)
        return self

if __name__ == "__main__":
    import sys
    sys.path.append("../")
    from det3.dataloarder.data import KittiData
    from PIL import Image
    data = KittiData('/usr/app/data/KITTI/dev/', '000000')
    calib, _, label, pc = data.read_data()
    bevimg = BEVImage(x_range=(0, 70), y_range=(-30,30), grid_size=(0.05, 0.05))
    bevimg.from_lidar(pc, scale=1)
    for obj in label.read_kitti_label_file().data:
        if obj.type == 'Pedestrian':
            bevimg.draw_box(obj, calib, bool_gt=False)
        print(obj)
    bevimg_img = Image.fromarray(bevimg.data)
    bevimg_img.save("lala.png")

