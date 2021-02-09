from matplotlib import pyplot as plt
from numpy import multiply, array, median, meshgrid, arange, zeros, dstack
from numpy.linalg import inv
from math import floor, ceil
from scipy.stats import mode
import plotly.graph_objects as go
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle

class Car:
    def __init__(self, image, depth, instance, camera, base_camera, car_index, depth_normalization_func=None):
        self.image = image
        self.depth = depth
        self.instance = instance
        if not depth_normalization_func:
            self.normalized_depth = depth - median(depth)
        else:
            self.normalized_depth = depth_normalization_func(self.depth, self.instance)
        self.camera = camera
        self.base_camera = base_camera
        self.car_index = car_index

    def plot(self, masked=True):
        if masked:
            fig, ax = plt.subplots(2, figsize=(35,20))
            i = self.instance == mode(self.instance.ravel())[0][0]
            ax[0].imshow(dstack((multiply(self.image[:,:,0], i), multiply(self.image[:,:,1],i), multiply(self.image[:,:,2],i))).astype(int))
            ax[1].imshow(multiply(self.depth, self.instance == mode(self.instance.ravel())[0][0]))
            ax[0].set_title('Image')
            ax[1].set_title('Depth')
        else:
            fig, ax = plt.subplots(3, figsize=(50,20))
            ax[0].imshow(self.image)
            ax[1].imshow(self.depth)
            ax[2].imshow(self.instance)
            ax[0].set_title('Image')
            ax[1].set_title('Depth')
            ax[2].set_title('Instance map')

    def plot_3d(self, masked=True):
        x, y = meshgrid(arange(self.depth.shape[1]), arange(self.depth.shape[0]))
        inv_camera = inv(self.camera)
        if masked:
            x, y = x[self.instance == mode(self.instance.ravel())[0][0]], y[self.instance == mode(self.instance.ravel())[0][0]]
            z = zeros((x.shape[0], 3))
            for a, (i, j) in enumerate(zip(x, y)):
                z[a] = inv_camera @ array([i, j, 1]) * self.depth[j, i]
            
            fig = go.Figure(data=go.Scatter3d(x=z[:,0].ravel(), y=z[:,1].ravel(), z=z[:,2].ravel(),
                                            mode='markers',
                                            marker=dict(size=1)        
                                            )
                            )

            x_range = [floor(z[:,0].min()), ceil(z[:,0].max())]
            y_range = [floor(z[:,1].min()), ceil(z[:,1].max())]
            z_range = [floor(z[:,2].min()), ceil(z[:,2].max())]
            
        else:
            z = zeros((x.shape[0], x.shape[1], 3)).astype(float)
            for i,j in zip(x.ravel(), y.ravel()):
                z[j, i] = inv_camera @ array([i, j, 1]) * self.depth[j, i]

            fig = go.Figure(data=go.Scatter3d(x=z[:,:,0].ravel(), y=z[:,:,1].ravel(), z=z[:,:,2].ravel(),
                                            mode='markers',
                                            marker=dict(size=1, color=self.instance.ravel(), colorscale='Viridis')        
                                            )
                            )
            
            x_range = [floor(z[:,:,0].min()), ceil(z[:,:,0].max())]
            y_range = [floor(z[:,:,1].min()), ceil(z[:,:,1].max())]
            z_range = [floor(z[:,:,2].min()), ceil(z[:,:,2].max())]

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