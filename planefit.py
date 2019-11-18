import numpy as np
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import chart_studio.plotly as py
import pandas as pd
import utilities

df = pd.read_pickle("./gcmt_all_earthquakes.pkl")
df = df[(df.lat > 50) & (df.lat < 75) & (df.lon < -130)]
data = np.row_stack([utilities.spherical2cart(utilities.row2mt(row).pos) for row in df.to_numpy()])

mn = np.min(data, axis=0)
mx = np.max(data, axis=0)
X,Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))
XX = X.flatten()
YY = Y.flatten()

# best-fit quadratic curve (2nd-order)
A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])

# evaluate it on a grid
Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2], C).reshape(X.shape)

trace1 = go.Scatter3d(
    x=data[:,0],
    y=data[:,1],
    z=data[:,2],
    mode='markers',
    marker=dict(size=4, color='red', line=dict(color='black', width=0.5), opacity=0.6)
)

trace3 = go.Surface(
    z=Z,
    x=X,
    y=Y,
    colorscale='RdBu',
    opacity=0.6
)

data_test2 = [trace1, trace3]

layout = go.Layout(title='2nd order surface')

fig = go.Figure(data=data_test2, layout=layout)
fig.show()
