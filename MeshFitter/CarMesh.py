import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import torch
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

class CarMesh:
    def __init__(self):
        self.icosphere = ico_sphere(4, device)
        self.icosphere.textures = TexturesVertex(torch.ones_like(self.icosphere.verts_packed())[None])

    def plot(self):
        R, T = look_at_view_transform(dist=-2.5, elev=0, azim=0)
        camera = OpenGLPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=(64,128), 
            blur_radius=0, 
            faces_per_pixel=1, 
            perspective_correct=False
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=camera
            )
        )

        image = renderer(self.icosphere, cameras=camera)

        plt.imshow(image.cpu().numpy()[0])

    def plot_3d(self):
        x,y,z = self.icosphere.verts_packed().cpu().numpy().T
        i,j,k = self.icosphere.faces_packed().cpu().numpy().T
        fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, colorscale=[[0, 'gold'], [1, 'magenta']], intensity=y)])
        return fig
