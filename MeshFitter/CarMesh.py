import numpy as np
import plotly.graph_objects as go
import torch
from pytorch3d.utils import ico_sphere

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

class CarMesh:
    def __init__(self):
        self.icosphere = ico_sphere(4, device)

    def plot(self):
        x,y,z = self.icosphere.verts_packed().cpu().numpy().T
        i,j,k = self.icosphere.faces_packed().cpu().numpy().T
        fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, colorscale=[[0, 'gold'], [1, 'magenta']], intensity=y)])
        return fig