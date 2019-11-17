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
print(df)
df = df[(df.lat > 50) & (df.lat < 75) & (df.lon < -100)]
print(df)
MomentTensors = [utilities.row2mt(row) for row in df.to_numpy()]

max_distance = 5000.
initial_point = (utilities.radius(55) - 100, 57.59, -166.69)

closest_tensor = MomentTensors[np.argmin([utilities.distance(initial_point, MT.pos) for MT in MomentTensors])]
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
            if utilities.distance(MT_accepted.pos, MT.pos) < max_distance:
                testing_tensors.append(MT)
                break
    for MT_testing in testing_tensors:
        testing_tensors_m0_sum = utilities.sum_m0(accepted_tensors[0:2] + [MT_testing])
        testing_tensor_sum_m0 = utilities.tensor_sum(accepted_tensors[0:2] + [MT_testing]).m0
        Cs = testing_tensor_sum_m0 / testing_tensors_m0_sum
        accepted_tensors.append(MT_testing) if Cs > 0.97 else rejected_tensors.append(MT_testing)
        print(Cs)

print(len(testing_tensors), len(accepted_tensors), len(rejected_tensors))
print(len(accepted_tensors) + len(rejected_tensors))
print(len(MomentTensors))


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
            llcrnrlat=45, llcrnrlon=-180, urcrnrlat=72, urcrnrlon=-100)
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
plt.show()
