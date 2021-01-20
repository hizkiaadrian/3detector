from matplotlib import pyplot as plt
import plotly.graph_objects as go
from numpy import multiply

class Car:
    def __init__(self, base_image_path, image, depth, instance, world_coordinates, car_index):
        self.base_image_path = base_image_path
        self.image = image
        self.depth = depth
        self.instance = instance
        self.world_coordinates = world_coordinates
        self.car_index = car_index

    def plot(self):
        fig, ax = plt.subplots(3, figsize=(50,20))
        ax[0].imshow(self.image)
        ax[1].imshow(self.depth)
        ax[2].imshow(self.instance)
        ax[0].set_title('Image')
        ax[1].set_title('Depth')
        ax[2].set_title('Instance map')

    def plot_3d(self, type=None, isolateCar=False):
        x = self.world_coordinates[:,:,0].ravel() if isolateCar else multiply(self.world_coordinates[:,:,0], self.instance >0).ravel()
        x = x[x != 0]
        y = self.world_coordinates[:,:,1].ravel() if isolateCar else multiply(self.world_coordinates[:,:,1], self.instance >0).ravel()
        y = y[y != 0]
        z = self.world_coordinates[:,:,2].ravel() if isolateCar else multiply(self.world_coordinates[:,:,2], self.instance >0).ravel()
        z = z[z != 0]

        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=3, color=z, colorscale='Sunset')
        )])

        if type=='full':
            fig.update_layout(
                    margin=dict(l=0, r=0, t=20,b=20),
                    scene = dict(
                        aspectmode='manual',
                        aspectratio=dict(x=2, y=2, z=5),
                        xaxis = dict(title='Width', range=[-20,20]),
                        yaxis = dict(title='Height', range=[-20,20]),
                        zaxis = dict(title='Depth', range=[0,100])
                    ),
                    height=800
                )
        else:
            fig.update_layout(
                margin=dict(l=0, r=0, t=20,b=20),
                scene = dict(
                    aspectmode='manual',
                    aspectratio=dict(x=3, y=1, z=10),
                    xaxis_title="Width",
                    yaxis_title="Height",
                    zaxis_title="Depth"
                ),
                height=800
            )

        fig.update_layout(scene_camera=dict(
                        eye=dict(x=1.25, y=1.25, z=1.25),
                        up=dict(x=0,y=-1,z=0),
                        center=dict(x=0,y=0,z=0)
                    ))

        return fig
