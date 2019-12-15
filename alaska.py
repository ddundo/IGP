import os
os.environ["PROJ_LIB"] = "C:\\Users\\Dundo\\Anaconda3\\Library\\share";  # quick fix
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import utilities
import numpy as np
import pandas as pd
from obspy.imaging.beachball import beach, beachball, mt2plane
from matplotlib.colors import Normalize
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.spatial import ConvexHull
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import plotly.graph_objs as go
from datetime import datetime
from scipy.interpolate import griddata


sns.set_style("white")
flatten = lambda l: [item for sublist in l for item in sublist]

df = pd.read_pickle("./gcmt_all_earthquakes.pkl")
df = df[(df.lat > 40) & (df.lat < 72) & (df.lon > -175) & (df.lon < -120)]
MomentTensors = [utilities.row2mt(row) for row in df.to_numpy()]

print('Total number of tensors: ', len(MomentTensors))

coords = [(mt.lon, mt.lat) for mt in MomentTensors]
depth = [mt.depth for mt in MomentTensors]

nzones = 7

zone1 = Polygon([(-172.038, 52.6893), (-171.792, 51.0717), (-166.011, 52.6026),
                 (-166.849, 53.4818), (-167.397, 53.1612), (-168., 53.), (-168.827, 52.5955)])
zone2 = Polygon([(-170.725, 50.556), (-165.744, 50.8138), (-166., 52.3845), (-168.944, 51.7633)])
zone3 = Polygon([(-170.583, 52.8952), (-169.112, 52.7031), (-165.596, 53.9039),
                 (-164.854, 53.8886), (-164.317, 53.9718), (-163.523, 54.4418),
                 (-165.569, 54.9218), (-170.543, 53.0608)])
zone4 = Polygon([(-164.762, 53.7826), (-164.384, 52.8621), (-160.777, 53.7364), (-156.562, 55.4317),
                 (-156.648, 55.8754), (-159.271, 55.2198), (-160.37, 55.4383)])
zone5 = Polygon([(-155.522, 56.3125), (-155.291, 55.4781), (-150.449, 57.2113),
                 (-153.696, 57.4252), (-154.608, 56.3192), (-155.416, 56.3854)])
zone6 = Polygon([(-154.185, 60.3392), (-155.695, 58.4186), (-154.661, 58.4981),
                 (-153.529, 59.1074), (-152.894, 59.0815),
                 (-152.401, 60.6241), (-153.509, 60.5644), (-153.562, 60.2332)])
zone7 = Polygon([(-150.564, 61.7916), (-150.616, 60.8065), (-149.6, 60.87), (-149.879, 61.685)])
# zone8 = Polygon([(-151.4, 62.), (-150.95, 62.3837), (-151.48, 62.8187), (-152.058, 62.1726)])
# zone10 = Polygon([(-131.373, 52.037), (-130.826, 49.6386), (-129.953, 49.4524), (-129.435, 48.1776),
#                   (-127.072, 48.8121), (-128.597, 52.2638)])

zone_tensors = [[] for _ in range(nzones)]

for i in range(len(MomentTensors)):
    if zone1.contains(Point(coords[i])):
        if depth[i] < 75:
            zone_tensors[0].append(MomentTensors[i])
    elif zone2.contains(Point(coords[i])):
        if depth[i] < 60:
            zone_tensors[1].append(MomentTensors[i])
    elif zone3.contains(Point(coords[i])):
        if 175 > depth[i] > 65:
            zone_tensors[2].append(MomentTensors[i])
    elif zone4.contains(Point(coords[i])):
        if depth[i] < 40:
            zone_tensors[3].append(MomentTensors[i])
    elif zone5.contains(Point(coords[i])):
        if depth[i] < 40:
            zone_tensors[4].append(MomentTensors[i])
    elif zone6.contains(Point(coords[i])):
        if depth[i] > 90:
            zone_tensors[5].append(MomentTensors[i])
    elif zone7.contains(Point(coords[i])):
        if depth[i] < 70:
            zone_tensors[6].append(MomentTensors[i])
    # elif zone8.contains(Point(coords[i])):
    #     if 150 > depth[i] > 70:
    #         zone_tensors[7].append(MomentTensors[i])
    # elif zone10.contains(Point(coords[i])):
    #     if depth[i] < 50:
    #         zone_tensors[8].append(MomentTensors[i])

for i in range(nzones):
    print('Zone ', i + 1)
    print('Number of tensors: ', len(zone_tensors[i]))
    print('Cs: ', utilities.seismic_consistency(zone_tensors[i]))

zone_tensors_coords_2D = [[(mt.lon, mt.lat) for mt in tensors] for tensors in zone_tensors]
# zone_tensors_coords = [[utilities.spherical2cart(mt.pos) for mt in tensors] for tensors in zone_tensors]

zone_hulls_2D = [ConvexHull(zone_coords) for zone_coords in zone_tensors_coords_2D]
# zone_hulls = [ConvexHull(zone_coords) for zone_coords in zone_tensors_coords]

sum_mts = [utilities.tensor_sum_normalized(tensors) for tensors in zone_tensors]

princ_axes = [utilities.princax(tensor) for tensor in sum_mts]
print(princ_axes)

nodal_planes = [[mt2plane(mt).strike, mt2plane(mt).dip, mt2plane(mt).rake] for mt in sum_mts]
print(nodal_planes)
strikes = [plane[0] for plane in nodal_planes]

# data = []

usgs = pd.read_pickle("usgs_alaska_1900-2019_extracted.p")
# usgs = usgs.sort_values('magnitude')
usgs_mws = usgs[['magnitude']].to_numpy().flatten()
# pos = usgs[['depth', 'latitude', 'longitude']].to_numpy()
# plt.hist([mt.mw for mt in MomentTensors], np.arange(4, 10, 1))
# plt.show()
# stop
usgs_lats = usgs[['latitude']].to_numpy().flatten()
usgs_lons = usgs[['longitude']].to_numpy().flatten()
usgs_depths = usgs[['depth']].to_numpy().flatten()
usgs_times = usgs[['time']].to_numpy().flatten()
usgs_m0s = utilities.mw2m0(usgs_mws)

velocities = []
invariants = []

zones_lats = flatten([[mt.lat for mt in tensors] for tensors in zone_tensors])
zones_lons = flatten([[mt.lon for mt in tensors] for tensors in zone_tensors])

for i in range(len(zone_tensors)):
    # X, Y, Z, area = utilities.planefit(zone_tensors[i])
    m0_sum = 0
    _, _, _, area = utilities.planefit(zone_tensors[i])
    mu = 3.3e10
    times = []
    for j in range(len(usgs_m0s)):
        if utilities.point_in_hull((usgs_lons[j], usgs_lats[j]), zone_hulls_2D[i]):
            m0_sum += usgs_m0s[j]
            times.append(usgs_times[j])

    t = pd.Timedelta(max(times) - min(times)).value / 3.154e+16

    print(m0_sum, area[0], t, len(times))

    v = utilities.dynecm2nm(m0_sum) * 1e3 / (mu * area[0] * t)
    if i == 2:
        v *= 3
    print(f'Velocity of zone {i + 1} is {v} mm/yr')
    for _ in range(len(zone_tensors[i])):
        velocities.append(v)

    times = [mt.date for mt in zone_tensors[i]]
    # print(max(times), min(times),
    #       (max(times) - min(times)).total_seconds())
    t = (max(times) - min(times)).total_seconds() / 3.154e+7
    tensor_sum = utilities.tensor_sum(zone_tensors[i])
    eps = utilities.dynecm2nm(tensor_sum.mt_e) / (2 * mu * area[0] * 80e3)
    J2 = utilities.second_invariant(eps)
    print('2nd invariant: ', J2)
    print(i + 1, 'volume', area[0] * 80 / 1e6)
    for _ in range(len(zone_tensors[i])):
        invariants.append(J2)
# stop
# xi = np.linspace(-172.6, -146.4, 30)
# yi = np.linspace(49.3, 62.8215)
# xi, yi = np.meshgrid(xi, yi)
#
# vi = griddata((zones_lons, zones_lats), invariants, (xi, yi), method='linear')
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.contourf(xi, yi, vi)
# plt.colorbar()
# plt.show()

#     points = np.row_stack([utilities.spherical2cart(mt.pos) for mt in zone_tensors[i]])
#
#     trace1 = go.Scatter3d(
#         x=-1 * points[:,0],
#         y=points[:,1],
#         z=points[:,2],
#         mode='markers',
#         marker=dict(size=4, color='red', line=dict(color='black', width=0.5), opacity=0.6)
#     )
#
#     trace3 = go.Surface(
#         z=Z,
#         x=-1 * X,
#         y=Y,
#         # colorscale='RdBu',
#         showscale=False,
#         opacity=0.6
#     )
#
#     data.append(trace1)
#     data.append(trace3)
#
# layout = go.Layout(title='2nd order surface')
# fig = go.Figure(data=data, layout=layout)
# fig.show()

# stop

# srop

# MAP PLOT
# textcolors = ['r', 'm', 'g', 'b']
# colors = ['r--', 'm--', 'g--', 'b--']

# m = Basemap(projection='cyl',llcrnrlat=45,urcrnrlat=75,\
#             llcrnrlon=-175,urcrnrlon=-120,resolution='h')
m = Basemap(  #width=12000000, height=9000000,
            rsphere=(6378137.00, 6356752.3142),
            resolution='h', area_thresh=100., projection='lcc',
            # lat_1=50., lat_2=60, lat_0=55, lon_0=-135.,
            lat_0=63, lon_0=-147.,
            llcrnrlat=45, llcrnrlon=-180, urcrnrlat=65, urcrnrlon=-100,
            epsg=3338)
m.drawcoastlines()
m.fillcontinents()
# draw parallels and meridians.
# m.drawparallels(np.linspace(45, 75, 31), linewidth=0.5, labels=[False, True, True, False])
# m.drawmeridians(np.linspace(-175, -120, 56), linewidth=0.5, labels=[True, False, True, False])
m.drawparallels(np.arange(-80., 81., 10.), color='black', linewidth=0.5, labels=[False, True, True, False])
m.drawmeridians(np.arange(-180., 181., 10.), color='black', linewidth=0.5, labels=[True, False, True, False])
m.drawmapboundary(fill_color='aqua')

# x, y = m([MT.lon for MT in MomentTensors], [MT.lat for MT in MomentTensors])
# focmecs = [MT.mt6 for MT in MomentTensors]

# zone_tensors_flat = flatten(zone_tensors)
# x, y = m([MT.lon for MT in zone_tensors_flat], [MT.lat for MT in zone_tensors_flat])
# focmecs = [MT.mt6 for MT in zone_tensors_flat]
# depths_flat = [MT.depth for MT in zone_tensors_flat]

x, y = m([np.mean([mt.lon for mt in tensors]) for tensors in zone_tensors],
         [np.mean([mt.lat for mt in tensors]) for tensors in zone_tensors])
focmecs = [mt.mt6 for mt in sum_mts]
depths_flat = [np.mean([mt.depth for mt in tensors]) for tensors in zone_tensors]

cmap = plt.cm.rainbow
norm = Normalize(vmin=np.min(depths_flat), vmax=np.max(depths_flat))

ax = plt.gca()
for i in range(len(focmecs)):
    b = beach(focmecs[i], xy=(x[i], y[i]), width=100000, facecolor=cmap(norm(depths_flat[i])), linewidth=1)
    b.set_zorder(10)
    ax.add_collection(b)
    # for j in range(len(distances)):
    #     ax.annotate(distances[j][i], xy=(x[i] + 0.1, y[i] - 0.1 + j * 0.1), color=textcolors[j])

for i in range(len(zone_tensors)):
    lons, lats = [mt.lon for mt in zone_tensors[i]], [mt.lat for mt in zone_tensors[i]]
    x, y = m(lons, lats)
    x = np.array(x)
    y = np.array(y)
    m.plot(x[zone_hulls_2D[i].vertices.tolist() + [zone_hulls_2D[i].vertices.tolist()[0]]],
           y[zone_hulls_2D[i].vertices.tolist() + [zone_hulls_2D[i].vertices.tolist()[0]]], '--', lw=3, label=f'Zone {i+1}')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
plt.colorbar(sm, orientation='horizontal')
plt.legend(loc='lower right')
plt.show()
