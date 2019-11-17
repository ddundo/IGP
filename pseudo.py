import pandas as pd
import numpy as np
import utilities
import os
import matplotlib.pyplot as plt
os.environ["PROJ_LIB"] = "C:\\Users\\Dundo\\Anaconda3\\Library\\share"; #fixr

df = pd.read_pickle('gcmt_lat5075_lon130180.pkl')
# for i, col in enumerate(df.columns):
#     print(i, col)
mat = df.to_numpy()

dates = mat[:, 0]

mts_6 = np.transpose(np.array([mat[:, 9], mat[:, 11], mat[:, 13], mat[:, 15], mat[:, 17], mat[:, 19]]))
mts_6_unc = np.transpose(np.array([mat[:, 10], mat[:, 12], mat[:, 14], mat[:, 16], mat[:, 18], mat[:, 20]]))
iexp = mat[:, 8]

mts = [utilities.MomentTensor(mt, exp) for mt, exp in zip(mts_6, iexp)]
mt = utilities.MomentTensor(mts_6[0], iexp[0])
eigval, eigvct = np.linalg.eigh(mt.mt)
print(eigval)

mt = utilities.MomentTensor(mts_6[0] + mts_6_unc[0], iexp[0])
eigval, eigvct = np.linalg.eigh(mt.mt)
print(eigval)

mt1 = mts_6_unc[0] * 0
mt1[0] = 0.01
print(mt1)
mt = utilities.MomentTensor(mts_6[0] + mt1, iexp[0])
eigval, eigvct = np.linalg.eigh(mt.mt)
print(eigval)

# for mts, exp in zip(mts_6_unc, iexp):
#     mt = utilities.MomentTensor(mts, exp)
#     norm = np.linalg.norm(mt.mt)
#     if norm > 10000:
#         print(norm)
#         print(mts)

mags = np.array([mt.mw for mt in mts])
print(np.mean(mags))

mts = [utilities.row2mt(row) for row in mat]
print(mts[0].m0)

print(mts[0].m0)
# for (i, j), value in np.ndenumerate(mts[0].mt_err):
#     for k in np.linspace(-value, value, 10):
#         mts[0].mt[i][j] += k
#         print(np.round(k, 4), np.round(mts[0].m0, 6))

m0_per = []
per_norms = []
pers = []
std = []

# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot as plt
#
# fig = plt.figure(figsize=plt.figaspect(1))
# ax = fig.add_subplot(111, projection='3d')
#
# coefs = (6356.752**(-2), 6378.137**(-2), 6378.137**(-2))
# rx, ry, rz = 1/np.sqrt(coefs)
#
# # u = np.linspace(0, 2 * np.pi, 100)
# # v = np.linspace(0, np.pi, 100)
# u = np.linspace(np.radians(-130), np.radians(-180), 25)
# v = np.linspace(np.radians(50), np.radians(75), 25)
#
# x = rx * np.outer(np.cos(u), np.sin(v))
# y = ry * np.outer(np.sin(u), np.sin(v))
# z = rz * np.outer(np.ones_like(u), np.cos(v))
#
# ax.plot_surface(x, y, z,  rstride=4, cstride=4, alpha=0.05)
#
# max_radius = max(rx, ry, rz)
# for axis in 'xyz':
#     getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))
#
# # color = [str(item.depth/255.) for item in mts]
#
# x, y, z, depth = [], [], [], []
# for mt in mts:
#     _x, _y, _z = utilities._spherical2cart(mt.pos)
#     x.append(_x); y.append(_y); z.append(_z); depth.append(mt.depth)
# im = ax.scatter(x, y, z, c=depth, cmap='jet')
# fig.colorbar(im, ax=ax)
# plt.show()
#
# htfrb
# ######## basemap
#
# from mpl_toolkits.basemap import Basemap
#
# fig=plt.figure()
# ax=fig.add_axes([.1,.1,.8,.8], projection='3d')  # This is the background axis
#
# for mt in mts:
#     x, y, z = utilities._spherical2cart(mt.pos)
#     ax.scatter(x, y, z)
#
#
# plt.setp(ax.get_xticklabels(),visible=False)
# plt.setp(ax.get_yticklabels(),visible=False)
# ax.patch.set_visible(False)
# ax.grid(False)
# ax.axis('off')
#
#
# ax2=fig.add_axes([.1,.1,.8,.8])
# m = Basemap(projection='ortho',lon_0=-105,lat_0=-25,resolution='l',ax=ax2)
# m.fillcontinents(color='coral', alpha=0.3)
#
# plt.show()

print(mts[0].mt)
print(mts[0].mt_err)
rate = mts[0].mt_err / mts[0].mt
print(np.round(rate * 100, 4))
print(mts[0].exp)

maxperturb = utilities.MomentTensor(mts[0].mt + mts[0].mt_err, mts[0].exp)
print((mts[0].m0 - maxperturb.m0) / mts[0].m0)
print(maxperturb.m0)


# errors = []
# for i in range(len(mts)):
#     errors.append(np.random.standard_normal(1)[0])
#
# print(errors)
#
# from scipy.stats import normaltest
#
# print(normaltest(errors))
# import matplotlib.pyplot as plt
# plt.hist(errors)
# plt.show()
#
# errors = np.random.standard_normal(867)
# print(normaltest(errors))
# plt.hist(errors)
# plt.show()

for k in [300]:#range(len(mts)):
    for _ in range(10000):
        per = np.zeros_like(mts[k].mt_err)
        for (i, j), value in np.ndenumerate(mts[k].mt_err):
            per[i][j] = np.random.normal(scale=value)
        new_mt = mts[k].mt + per
        mt_per = utilities.MomentTensor(new_mt, mts[k].exp)
        m0_per.append(mt_per.m0)
        # per_norms.append(np.linalg.norm(per))
        # pers.append(per)
        std.append(np.abs(mts[k].m0 - mt_per.m0))

print(np.mean(m0_per))
print(np.min(m0_per), np.max(m0_per))
# print(np.mean(per_norms))
print(np.mean(std))
#
# plt.hist(m0_per)
# plt.show()

import plotly.graph_objects as go

coefs = (6356.752**(-2), 6378.137**(-2), 6378.137**(-2))
rx, ry, rz = 1/np.sqrt(coefs)

u = np.linspace(np.radians(-130), np.radians(-180), 25)
v = np.linspace(np.radians(50), np.radians(75), 25)

x1 = rx * np.outer(np.cos(u), np.sin(v))
y1 = ry * np.outer(np.sin(u), np.sin(v))
z1 = rz * np.outer(np.ones_like(u), np.cos(v))
#
# ax.plot_surface(x, y, z,  rstride=4, cstride=4, alpha=0.05)
#
# max_radius = max(rx, ry, rz)
# for axis in 'xyz':
#     getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

# color = [str(item.depth/255.) for item in mts]

x, y, z, depth = [], [], [], []
for mt in mts:
    _x, _y, _z = utilities._spherical2cart(mt.pos)
    x.append(_x); y.append(_y); z.append(_z); depth.append(mt.depth)

fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=3,
        color=depth,            # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
), go.Surface(
        x=x1,
        y=y1,
        z=z1, opacity=0.15
    )])

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
