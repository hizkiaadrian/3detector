import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pytorch3d.utils import ico_sphere
from pytorch3d.renderer import (
    TexturesVertex,
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader
)
from plotly.graph_objects import Figure, Scatter3d
from math import floor, ceil

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader
        
    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        image = self.shader(fragments, meshes_world, **kwargs)
        return image, fragments.zbuf

class CarMesh:
    def __init__(self, K, mesh_color=(0.7,0,0)):
        '''
        K is a 3x3 matrix, needs to be expanded to 4 by 4
        mesh_color is a tuple of 3 floats from 0 to 1 representing RGB values respectively
        '''
        self.mesh = ico_sphere(2, device)

        tex_shape = torch.Size([self.mesh.verts_padded().shape[0], self.mesh.verts_padded().shape[1], 1])
        self.mesh.textures = TexturesVertex(torch.cat((torch.full(tex_shape, mesh_color[0], device=device),
                                                 torch.full(tex_shape, mesh_color[1], device=device),
                                                 torch.full(tex_shape, mesh_color[2], device=device)),2))

        self.camera = PerspectiveCameras(device = device, 
                                         focal_length = ((K[0,0], K[1,1]),), 
                                         principal_point = ((K[0,2], K[1,2]),), 
                                         image_size = ((128, 64),)
                                        )
        
        raster_settings = RasterizationSettings(
            image_size=(64,128), 
            blur_radius=np.log(1. / 1e-4 - 1.)*1e-4,
            faces_per_pixel=1, 
            perspective_correct=True
        )

        self.renderer = MeshRendererWithDepth(
            rasterizer=MeshRasterizer(
                cameras=self.camera,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=self.camera
            )
        )

        self.render()

    def render(self):
        self.image, self.depth = self.renderer(self.mesh, cameras=self.camera)

    def plot(self):
        fig, ax = plt.subplots(2, figsize=(10,10))
        ax[0].imshow(self.image.cpu().numpy()[0])
        ax[0].set_title("Image")
        ax[1].imshow(self.depth.cpu().numpy()[0])
        ax[1].set_title("Depth")

    def plot_3d(self):
        z = self.mesh.verts_packed().cpu().numpy()

        fig = Figure(data=Scatter3d(x=z[:,0].ravel(), y=z[:,1].ravel(), z=z[:,2].ravel(),
                                    mode='markers',
                                    marker=dict(size=1),
                                    showlegend=False
                                )
                    )

        zfar = self.camera.transform_points(torch.Tensor([[0,0,z[:,2].max() + 5]]).to(device)).cpu().numpy()[0,2]
        corners = [[-1,1, zfar],[-1,-1,zfar],[1,-1,zfar],[1,1,zfar]]
        cp = [self.camera.get_full_projection_transform().inverse()
                    .transform_points(torch.Tensor([corner]).to(device)).cpu().numpy()[0] for corner in corners]

        corner_points = np.block([
            [cp[0]],
            [cp[1]], 
            [cp[2]], 
            [cp[3]]]
        )

        for corner_pt in corner_points:
            fig.add_trace(_draw_line([0,0,0], corner_pt))

        fig.add_trace(_draw_line(corner_points[0], corner_points[1]))
        fig.add_trace(_draw_line(corner_points[0], corner_points[3]))
        fig.add_trace(_draw_line(corner_points[1], corner_points[2]))
        fig.add_trace(_draw_line(corner_points[2], corner_points[3]))
        
        x_range = [floor(min([0, z[:,0].min(), corner_points[:,0].min()])), ceil(max([0, z[:,0].max(), corner_points[:,0].max()]))]
        y_range = [floor(min([0, z[:,1].min(), corner_points[:,1].min()])), ceil(max([0, z[:,1].max(), corner_points[:,1].max()]))]
        z_range = [-1, ceil(corner_points[:,2].max())]

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

def _draw_line(pt1, pt2):
    return Scatter3d(x=[pt1[0], pt2[0]],
                     y=[pt1[1], pt2[1]],
                     z=[pt1[2], pt2[2]],
                     marker=dict(size=2, color='black'),
                     showlegend=False
    )
