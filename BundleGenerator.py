import pykitti
from os import listdir
from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from math import floor, ceil
from os.path import exists

class DataBundle:
    def __init__(self, img_path, camera):
        self.img_path = img_path
        self.camera = camera
        self.image = cv2.imread(img_path)
        self.depth = np.load(img_path.replace('data', 'depth').replace('jpg','npz'))['depth']
        self.boxes = np.load(img_path.replace('image', 'panoptic').replace('jpg','npz'))['boxes']
        self.instances = np.load(img_path.replace('image', 'panoptic').replace('jpg','npz'))['instances']

        self._resize()

    def plot(self):
        fig, ax = plt.subplots(3, figsize=(70,30))
        ax[0].imshow(self.image)

        for box in self.boxes:
            rect = Rectangle((box[1], box[0]), box[3]-box[1], box[2]-box[0], edgecolor='r', facecolor='none')
            ax[0].add_patch(rect)

        ax[1].imshow(self.depth)
        ax[2].imshow(self.instances)

        plt.show()

    def _resize(self):
        width_ratio = self.depth.shape[1] / self.image.shape[1]
        height_ratio = self.depth.shape[0] / self.image.shape[0]

        self.image = cv2.resize(
            self.image, 
            (self.depth.shape[1], self.depth.shape[0]), 
            interpolation=cv2.INTER_NEAREST)
        self.instances = cv2.resize(
            self.instances, 
            (self.depth.shape[1], self.depth.shape[0]),
            interpolation=cv2.INTER_NEAREST)
        
        self.boxes = [
            [floor(box[0]*height_ratio), floor(box[1]*width_ratio), ceil(box[2]*height_ratio), ceil(box[3]*width_ratio)] 
            for box in self.boxes
            ]

        transformation_matrix = np.array([
                                    [width_ratio, 0, 0],
                                    [0, height_ratio, 0],
                                    [0, 0, 1]
                                ])
        self.camera = transformation_matrix @ self.camera
        
class BundleGenerator:
    def __init__(self, base_dir):
        self.dates = sorted(listdir(base_dir))
        self.base_dir = base_dir

    def load(self,date=None):
        if date is None:
            return self._yield_images()

        elif date in self.dates:
            return self._yield_images(date)

        else:
            raise ValueError

    def _yield_images(self, date = None):
        if not date:
            for date in self.dates:
                drives = sorted(
                    list(
                        map(
                            lambda x: x.split('_')[-2],
                            filter(lambda x: not x.endswith('txt'), listdir(f'{self.base_dir}/{date}'))
                        )
                    )
                )
                camera = pykitti.raw(self.base_dir, date, drives[0]).calib.K_cam2
                
                img_paths = sorted(
                    sum(
                        map(
                            lambda x: glob(
                                f'{self.base_dir}/{date}/{date}_drive_{x}_sync/image_02/data/*.jpg'),
                                drives
                            ), []
                    )
                )
                
                for img_path in img_paths:
                    try:
                        if not exists(img_path.replace('data', 'depth').replace('jpg','npz')):
                            raise FileNotFoundError("Depth file not found")
                        yield DataBundle(img_path, camera)
                    except:
                        print(f"Error with {img_path}")

        else:
            drives = sorted(
                list(
                    map(
                        lambda x: x.split('_')[-2],
                        filter(lambda x: not x.endswith('txt'), listdir(f'{self.base_dir}/{date}'))
                    )
                )
            )
            camera = pykitti.raw(self.base_dir, date, drives[0]).calib.K_cam2
            
            img_paths = sorted(
                sum(
                    map(
                        lambda x: glob(
                            f'{self.base_dir}/{date}/{date}_drive_{x}_sync/image_02/data/*.jpg'),
                            drives
                        ), []
                )
            )
            
            for img_path in img_paths:
                try:
                    if not exists(img_path.replace('data', 'depth').replace('jpg','npz')):
                        raise FileNotFoundError("Depth file not found")
                    yield DataBundle(img_path, camera)
                except FileNotFoundError:
                    continue