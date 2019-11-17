import pandas as pd
import utilities
import numpy as np
import os
os.environ["PROJ_LIB"] = "C:\\Users\\Dundo\\Anaconda3\\Library\\share"
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# Extracting Alaskan earthquakes
df = pd.read_pickle("./gcmt_all_earthquakes.pkl")
df = df[df.lat > 50]
df = df[df.lat < 75]
df = df[df.lon < -130]

mat = df.to_numpy()
mts_6 = np.transpose(np.array([mat[:, 9], mat[:, 11], mat[:, 13], mat[:, 15], mat[:, 17], mat[:, 19]]))
iexp = mat[:, 8]
lat = mat[:, 1]
lon = mat[:, 2]
depth = mat[:, 3]

MomentTensors = [utilities.MomentTensor(mts_6[i, :], iexp[i], lon[i], lat[i], depth[i]) for i in range(len(mts_6))]

max_distance = 200.
initial_point = (-150.96, 55, 10)

closest_tensor = MomentTensors[np.argmin([utilities.haversine(initial_point, MT.pos) for MT in MomentTensors])]
print(closest_tensor.pos)

accepted_tensors = [closest_tensor]
rejected_tensors = []

# while len(accepted_tensors) + len(rejected_tensors) < len(MomentTensors):
for _ in range(10):
    testing_tensors = []
    for MT in MomentTensors:
        if MT in rejected_tensors or MT in accepted_tensors:
            continue
        for MT_accepted in accepted_tensors:
            if utilities.haversine(MT_accepted.pos, MT.pos) < max_distance and MT.depth < 25:
                testing_tensors.append(MT)
                break
    for MT_testing in testing_tensors:
        testing_tensors_m0_sum = utilities.sum_m0(accepted_tensors + [MT_testing])
        testing_tensor_sum_m0 = utilities.m0_of_summed_tensor(accepted_tensors + [MT_testing])
        Cs = testing_tensor_sum_m0 / testing_tensors_m0_sum
        accepted_tensors.append(MT_testing) if Cs > 0.9 else rejected_tensors.append(MT_testing)
        print(Cs)

# print(len(testing_tensors), len(accepted_tensors), len(rejected_tensors))
# print(len(accepted_tensors) + len(rejected_tensors))
# print(len(MomentTensors))


import math
# import matplotlib.patches as patches
# import pylab
# pp=[(-0.500000050000005, -0.5), (-0.499999950000005, 0.5), (-0.500000100000005, -1.0), (-0.49999990000000505, 1.0), (0.500000050000005, -0.5), (-1.0000000250000025, -0.5), (1.0000000250000025, -0.5), (0.499999950000005, 0.5), (-0.9999999750000024, 0.5), (0.9999999750000024, 0.5), (0.500000100000005, -1.0), (0.49999990000000505, 1.0), (-1.0, 0.0), (-0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-0.500000050000005, -0.5)]
# # compute centroid
# cent=(sum([p[0] for p in pp])/len(pp),sum([p[1] for p in pp])/len(pp))
# # sort by polar angle
# pp.sort(key=lambda p: math.atan2(p[1]-cent[1],p[0]-cent[0]))
# # plot points
# pylab.scatter([p[0] for p in pp],[p[1] for p in pp])
# # plot polyline
# pylab.gca().add_patch(patches.Polygon(pp,closed=False,fill=False))
# pylab.grid()
# pylab.show()


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
            lat_1=55., lat_2=65, lat_0=62, lon_0=-158.,
            llcrnrlat=45, llcrnrlon=-180, urcrnrlat=72, urcrnrlon=-115)
m.drawcoastlines()
m.drawparallels(np.arange(-80., 81., 10.), labels=[False, True, True, False])
m.drawmeridians(np.arange(-180., 181., 20.), labels=[True, False, True, False])
draw_screen_poly([MT.lon for MT in accepted_tensors], [MT.lat for MT in accepted_tensors], m)
x, y = m([MT.lon for MT in accepted_tensors], [MT.lat for MT in accepted_tensors])
m.scatter(x, y, marker='o', color='r', alpha=0.5)
closest_x, closest_y = m(closest_tensor.lon, closest_tensor.lat)
m.scatter(closest_x, closest_y, marker='D')
plt.show()
