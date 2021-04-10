from matplotlib import pyplot as plt
from numpy import multiply, array, median, meshgrid, arange, zeros, dstack, vectorize
from numpy.linalg import inv
from math import floor, ceil
from scipy.stats import mode
from plotly.graph_objects import Figure, Scatter3d
import torch
from torch.utils.data import Dataset
import os
import pickle

def _get_instance_num(instance):
    return mode(instance[instance != 0])[0][0]

class Car:
    def __init__(self, image, depth, instance, camera, base_camera, car_index, depth_normalization_func):
        self.image = image
        self.depth = depth
        self.instance = vectorize(lambda i: 1 if i == _get_instance_num(instance) else 0)(instance)
        self.normalized_depth = depth_normalization_func(self.depth, self.instance)
        self.camera = camera
        self.base_camera = base_camera
        self.car_index = car_index

    def get_depth_map(self):
        depth = self.depth.copy()
        depth[self.instance != 1] = -1

        return depth

    def plot(self, masked=True, figsize=None):
        if masked:
            fig, ax = plt.subplots(2, figsize=(35,20) if not figsize else figsize)
            i = self.instance == mode(self.instance.ravel())[0][0]
            ax[0].imshow(dstack((multiply(self.image[:,:,0], i), multiply(self.image[:,:,1],i), multiply(self.image[:,:,2],i))).astype(int))
            ax[1].imshow(self.get_depth_map())
            ax[0].set_title('Image')
            ax[1].set_title('Depth')
        else:
            fig, ax = plt.subplots(3, figsize=(50,20) if not figsize else figsize)
            ax[0].imshow(self.image)
            ax[1].imshow(self.depth)
            ax[2].imshow(self.instance)
            ax[0].set_title('Image')
            ax[1].set_title('Depth')
            ax[2].set_title('Instance map')

    def get_point_clouds(self):
        x, y = meshgrid(arange(self.depth.shape[1]), arange(self.depth.shape[0]))
        inv_camera = inv(self.camera)
        x, y = x[self.instance == mode(self.instance.ravel())[0][0]], y[self.instance == mode(self.instance.ravel())[0][0]]
        z = zeros((x.shape[0], 3))
        for a, (i, j) in enumerate(zip(x, y)):
            z[a] = inv_camera @ array([i, j, 1]) * self.depth[j, i]

        return z

    def plot_3d(self):
        z = self.get_point_clouds()

        fig = Figure(data=Scatter3d(x=z[:,0].ravel(), y=z[:,1].ravel(), z=z[:,2].ravel(),
                                        mode='markers',
                                        marker=dict(size=1)        
                                        )
                        )

        x_range = [floor(z[:,0].min()), ceil(z[:,0].max())]
        y_range = [floor(z[:,1].min()), ceil(z[:,1].max())]
        z_range = [floor(z[:,2].min()), ceil(z[:,2].max())]

        x_len = x_range[1] - x_range[0]
        y_len = y_range[1] - y_range[0]
        z_len = z_range[1] - z_range[0]
        base = min([x_len, y_len, z_len])
        
        fig.update_layout(
            scene = dict(
                aspectmode='manual', aspectratio=dict(x=x_len/base, y=y_len/base, z=z_len/base),
                xaxis = dict(range=x_range),
                yaxis = dict(range=y_range),
                zaxis = dict(range=z_range)
            )
        )

        return fig

    def get_box(self):
        ref_height, ref_width = self.image.shape[:2]

        original_width = round(ref_width * self.base_camera[0,0] / self.camera[0,0])
        original_height = round(ref_height * self.base_camera[1,1] / self.camera[1,1])

        top_left_vertex_x = round(self.base_camera[0, 2] - (self.camera[0, 2] * original_width) / ref_width)
        top_left_vertex_y = round(self.base_camera[1, 2] - (self.camera[1, 2] * original_height) / ref_height)

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