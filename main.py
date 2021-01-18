import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class DataBundle:
    def __init__(self, img_path):
        self.image = cv2.imread(img_path)
        self.depth = np.load(img_path.replace('data', 'depth').replace('jpg','npz'))['depth']
        self.boxes = np.load(img_path.replace('image', 'panoptic').replace('jpg','npz'))['boxes']
        self.instances = np.load(img_path.replace('image', 'panoptic').replace('jpg','npz'))['instances']
        self.img_path = img_path

    def plot_image(self):
        fig, ax = plt.subplots(1, figsize=(20,30))

        ax.imshow(self.image)

        for box in self.boxes:
            rect = Rectangle((box[1], box[0]), box[3]-box[1], box[2]-box[0], edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        plt.show()

    def plot(self):
        fig, ax = plt.subplots(3, figsize=(70,30))
        ax[0].imshow(self.image)

        for box in self.boxes:
            rect = Rectangle((box[1], box[0]), box[3]-box[1], box[2]-box[0], edgecolor='r', facecolor='none')
            ax[0].add_patch(rect)

        ax[1].imshow(self.depth)
        ax[2].imshow(self.instances)

        plt.show()

kitti_path = '/scratch/local/hdd/hizkia/kitti'

img_path = glob.glob(f'{kitti_path}/2011_09_26/*/image_02/data/*.jpg')

ip = img_path[0]
db = DataBundle(ip)

n = input()