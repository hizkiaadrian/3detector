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
from pytorch3d.vis.plotly_vis import plot_scene, AxisArgs

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
    def __init__(self, K):
        '''
        K is a 3x3 matrix, needs to be expanded to 4 by 4
        '''
        self.icosphere = ico_sphere(4, device)
        self.icosphere.textures = TexturesVertex(torch.full(self.icosphere.verts_packed().shape, 0.5, device=device)[None])

        K_matrix = np.block([
            [
                [K, np.zeros((3,1))],
                [np.zeros((1,3)), 1]
            ]
        ])
        self.camera = PerspectiveCameras(device=device, K=K_matrix)
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

        self.image, self.depth = self.renderer(self.icosphere, cameras=self.camera)

    def plot(self):
        fig, ax = plt.subplots(2, figsize=(10,10))
        ax[0].imshow(self.image.cpu().numpy()[0])
        ax[0].set_title("Image")
        ax[1].imshow(self.depth.cpu().numpy()[0])
        ax[1].set_title("Depth")

    def plot_3d(self):
        fig = plot_scene({
            "World view": {
                "mesh": self.icosphere,
                "camera": self.camera
            }
        },
        xaxis={"backgroundcolor":"rgb(200, 200, 230)"},
        yaxis={"backgroundcolor":"rgb(230, 200, 200)"},
        zaxis={"backgroundcolor":"rgb(200, 230, 200)"}, 
        axis_args=AxisArgs(showgrid=True, showticklabels=True),
        viewpoint_cameras=self.camera
        )

        return fig
