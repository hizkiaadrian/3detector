from matplotlib import pyplot as plt
from numpy import multiply, array, median
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle

class Car:
    def __init__(self, image, depth, instance, camera, base_camera, car_index):
        self.image = image
        self.depth = depth
        self.normalized_depth = depth - median(depth)
        self.instance = instance
        self.camera = camera
        self.base_camera = base_camera
        self.car_index = car_index

    def plot(self):
        fig, ax = plt.subplots(3, figsize=(50,20))
        ax[0].imshow(self.image)
        ax[1].imshow(self.depth)
        ax[2].imshow(self.instance)
        ax[0].set_title('Image')
        ax[1].set_title('Depth')
        ax[2].set_title('Instance map')

    def get_box(self):
        ref_height, ref_width = self.image.shape[:2]

        original_width = round(ref_width * self.base_camera[0,0] / self.camera[0,0])
        original_height = round(ref_height * self.base_camera[1,1] / self.camera[1,1])

        top_left_vertex_x = round(base_camera[0, 2] - (self.camera[0, 2] * original_width) / ref_width)
        top_left_vertex_y = round(base_camera[1, 2] - (self.camera[1, 2] * original_height) / ref_height)

        return [top_left_vertex_y, top_left_vertex_x, top_left_vertex_y + original_height, top_left_vertex_x + original_width]

class CarDataset(Dataset):
    def __init__(self, base_path):
        self.base_path = base_path
        
    def __len__(self):
        return len(os.listdir(self.base_path))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        with open(f'{self.base_path}/{idx}.pkl','rb') as input:
            car = pickle.load(input)
            
        return car