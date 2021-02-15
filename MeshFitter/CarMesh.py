import numpy as np
import plotly.graph_objects as go

class CarMesh:
    def __init__(self, center = 0, radius = 5):
        self.vertices = np.array([
            [radius * np.sin(azimuth) * np.cos(theta), 
             radius * np.sin(azimuth) * np.sin(theta),
             radius * np.cos(azimuth)] 
             for azimuth in np.linspace(0,np.pi,60) for theta in np.linspace(0,2*np.pi,60)]
            )

    def plot(self):
        x,y,z = self.vertices.T
        fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, alphahull=0, colorscale=[[0, 'gold'], [1, 'magenta']], intensity=np.linspace(0,1,num=x.shape[0]))])
        return fig