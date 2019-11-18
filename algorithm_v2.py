import pandas as pd
import utilities
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
os.environ["PROJ_LIB"] = "C:\\Users\\Dundo\\Anaconda3\\Library\\share"
from mpl_toolkits.basemap import Basemap

df = pd.read_pickle("./gcmt_all_earthquakes.pkl")
df.loc[df['lon'] < 0, 'lon'] += 360
df = df[(df.lat > 45) & (df.lat < 72) & (df.lon > 170) & (df.lon < 235)]
df.loc[df['lon'] > 180, 'lon'] -= 360
MomentTensors = [utilities.row2mt(row) for row in df.to_numpy()]

axes = [utilities.princax(MomentTensors[i]) for i in range(len(MomentTensors))]

tdip = [ax[0][1] for ax in axes]
bdip = [ax[1][1] for ax in axes]
pdip = [ax[2][1] for ax in axes]

x_plot, y_plot = [], []

for i in range(len(tdip)):
    x, y = utilities.kaverina(tdip[i], bdip[i], pdip[i])
    x_plot.append(x)
    y_plot.append(y)

# fig = plt.figure()
# plt.axes().set_aspect('equal')
# plt.scatter(x_plot, y_plot, marker='x', alpha=0.7)
# plt.show()

max_distance = 150.
initial_point = (utilities.radius(51.720770) - 30, 51.720770, -173.733814)

closest_tensor = MomentTensors[np.argmin([utilities.distance(initial_point, MT.pos) for MT in MomentTensors])]
print(closest_tensor.pos)

accepted_tensors = [closest_tensor]
rejected_tensors = []

# while len(accepted_tensors) + len(rejected_tensors) < len(MomentTensors):
for _ in range(100):
    testing_tensors = []
    for MT in MomentTensors:
        if MT in rejected_tensors or MT in accepted_tensors:
            continue
        for MT_accepted in accepted_tensors:
            if utilities.distance(MT_accepted.pos, MT.pos) < max_distance:
                testing_tensors.append(MT)
                break
    if len(testing_tensors) == 0:
        break
    for MT_testing in testing_tensors:
        testing_tensors_m0_sum = utilities.sum_m0(accepted_tensors[0:1] + [MT_testing])
        testing_tensor_sum_m0 = utilities.tensor_sum(accepted_tensors[0:1] + [MT_testing]).m0
        Cs = testing_tensor_sum_m0 / testing_tensors_m0_sum
        accepted_tensors.append(MT_testing) if Cs > 0.99 else rejected_tensors.append(MT_testing)
        print(Cs)

print(len(testing_tensors), len(accepted_tensors), len(rejected_tensors))
print(len(accepted_tensors) + len(rejected_tensors))
print(len(MomentTensors))

depth = [MT.depth for MT in accepted_tensors]
print(depth)
print(np.mean(depth))

axes = np.array([utilities.princax(accepted_tensors[i]) for i in range(len(accepted_tensors))])
tdip, bdip, pdip = axes[:, 0, 1], axes[:, 1, 1], axes[:, 2, 1]

x_plot, y_plot = utilities.kaverina(tdip, bdip, pdip)

fig = plt.figure()
plt.axes().set_aspect('equal')

B = np.degrees(np.arcsin(np.sqrt(np.linspace(0, 1, 51) * 0.14645)))
P = np.degrees(np.arcsin(np.sqrt((1 - np.linspace(0, 1, 51)) * 0.14645)))

X, Y = utilities.kaverina(67.5, B, P)
plt.plot(X, Y, color='grey', linewidth=1)
X, Y = utilities.kaverina(P, B, 67.5)
plt.plot(X, Y, color='grey', linewidth=1)
X, Y = utilities.kaverina(P, 67.5, B)
plt.plot(X, Y, color='grey', linewidth=1)

T = np.degrees(np.arcsin(np.sqrt(np.linspace(0, 1, 101))))
P = np.degrees(np.arcsin(np.sqrt(1 - (np.linspace(0, 1, 101)))))

X, Y = utilities.kaverina(T, 0., P)
plt.plot(X, Y, color='black', linewidth=2)
X, Y = utilities.kaverina(P, T, 0.)
plt.plot(X, Y, color='black', linewidth=2)
X, Y = utilities.kaverina(0., P, T)
plt.plot(X, Y, color='black', linewidth=2)

plt.scatter(x_plot, y_plot, marker='x', alpha=0.7)

x_plot, y_plot = utilities.kaverina(tdip[0], bdip[0], pdip[0])
plt.scatter(x_plot, y_plot, marker='D', alpha=0.7)
plt.show()

def draw_screen_poly(lons, lats, m):
    xx, yy = m(lons, lats)
    xy = zip(xx, yy)
    pp = [(x, y) for x, y in zip(xx, yy)]
    cent = (sum([p[0] for p in pp]) / len(pp), sum([p[1] for p in pp]) / len(pp))
    pp.sort(key=lambda p: math.atan2(p[1] - cent[1], p[0] - cent[0]))
    poly = Polygon(pp, facecolor='blue', alpha=0.4)
    plt.gca().add_patch(poly)


m = Basemap(  #width=12000000, height=9000000,
            rsphere=(6378137.00, 6356752.3142),
            resolution='i', area_thresh=100., projection='lcc',
            lat_1=30., lat_2=50, lat_0=40, lon_0=-135.,
            llcrnrlat=35, llcrnrlon=175, urcrnrlat=72, urcrnrlon=-90)
m.drawcoastlines()
m.drawparallels(np.arange(-80., 81., 10.), labels=[False, True, True, False])
m.drawmeridians(np.arange(-180., 181., 20.), labels=[True, False, True, False])
draw_screen_poly([MT.lon for MT in accepted_tensors], [MT.lat for MT in accepted_tensors], m)
x, y = m([MT.lon for MT in accepted_tensors], [MT.lat for MT in accepted_tensors])
m.scatter(x, y, marker='o', color='g', alpha=0.5)
closest_x, closest_y = m(closest_tensor.lon, closest_tensor.lat)
m.scatter(closest_x, closest_y, marker='D')
x, y = m([MT.lon for MT in rejected_tensors], [MT.lat for MT in rejected_tensors])
m.scatter(x, y, marker='o', color='r', alpha=0.5)
# x, y = m([MT.lon for MT in MomentTensors], [MT.lat for MT in MomentTensors])
# depth = [MT.depth for MT in MomentTensors]
# m.scatter(x, y, marker='o', c=depth, alpha=0.5)
plt.show()
