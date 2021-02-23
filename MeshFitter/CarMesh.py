import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pytorch3d.utils import ico_sphere
from pytorch3d.renderer import (
    TexturesVertex,
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader
)

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
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf

class CarMesh:
    def __init__(self):
        self.icosphere = ico_sphere(4, device)
        self.icosphere.textures = TexturesVertex(torch.ones_like(self.icosphere.verts_packed())[None])

        R, T = look_at_view_transform(dist=-2.5, elev=0, azim=0)
        camera = OpenGLPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=(64,128), 
            blur_radius=0, 
            faces_per_pixel=1, 
            perspective_correct=False
        )

        self.renderer = MeshRendererWithDepth(
            rasterizer=MeshRasterizer(
                cameras=camera,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=camera
            )
        )

        self.image, self.depth = self.renderer(self.icosphere, cameras=camera)

    def plot(self):
        fig, ax = plt.subplots(2, figsize=(10,10))
        ax[0].imshow(self.image.cpu().numpy()[0])
        ax[0].set_title("Image")
        ax[1].imshow(self.depth.cpu().numpy()[0])
        ax[1].set_title("Depth")

    def plot_3d(self):
        x,y,z = self.icosphere.verts_packed().cpu().numpy().T
        i,j,k = self.icosphere.faces_packed().cpu().numpy().T
        fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, colorscale=[[0, 'gold'], [1, 'magenta']], intensity=y)])
        return fig