import os
os.environ["PROJ_LIB"] = "C:\\Users\\Dundo\\Anaconda3\\Library\\share";  # quick fix
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import utilities
import numpy as np
import pandas as pd
from obspy.imaging.beachball import beach
from matplotlib.colors import Normalize


df = pd.read_pickle("./gcmt_all_earthquakes.pkl")
df = df[(df.lat > 40) & (df.lat < 72) & (df.lon > -180) & (df.lon < -110)]
MomentTensors = [utilities.row2mt(row) for row in df.to_numpy()]

# choy = pd.read_pickle("./choy_alaska.p").to_numpy()

m = Basemap(projection='cyl',llcrnrlat=45,urcrnrlat=75,\
            llcrnrlon=-175,urcrnrlon=-120,resolution='h')
m.drawcoastlines()
m.fillcontinents()
m.drawparallels(np.arange(-45.,75.,30.), labels=[False, True, True, False])
m.drawmeridians(np.arange(-175.,-120.,55.), labels=[True, False, True, False])
m.drawmapboundary(fill_color='aqua')
x, y = m([MT.lon for MT in MomentTensors], [MT.lat for MT in MomentTensors])
depth = [MT.depth for MT in MomentTensors] #+ choy[:, 4].tolist()
focmecs = [MT.mt6 for MT in MomentTensors] #+ choy[:, -3:].tolist()

cmap = plt.cm.rainbow
norm = Normalize(vmin=np.min(depth), vmax=np.max(depth))

ax = plt.gca()
for i in range(len(focmecs)):
    b = beach(focmecs[i], xy=(x[i], y[i]), width=0.2, facecolor=cmap(norm(depth[i])), linewidth=1)
    b.set_zorder(10)
    ax.add_collection(b)
# m.scatter(x, y, marker='o', c=depth, edgecolors='white)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
plt.colorbar(sm)
plt.show()

stop

df = pd.read_pickle("./usgs_alaska_1900-2019.p")
new_df = df[['id', 'time', 'location', 'latitude', 'longitude', 'depth', 'magnitude', 'magtype', 'url']].copy()
new_df.reset_index(inplace=True, drop=True)
MomentTensors = new_df.to_numpy()[::3]

# print(type(MomentTensors[0, 1]))
# stop

m = Basemap(  #width=12000000, height=9000000,
            rsphere=(6378137.00, 6356752.3142),
            resolution='h', area_thresh=100., projection='lcc',
            # lat_1=50., lat_2=60, lat_0=55, lon_0=-135.,
            lat_0=63, lon_0=-147.,
            llcrnrlat=45, llcrnrlon=-180, urcrnrlat=65, urcrnrlon=-100,
            epsg=3338)
m.drawcoastlines()
m.fillcontinents()
m.drawmapboundary(fill_color='aqua')
# m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels=20000, verbose=True)
m.drawparallels(np.arange(-80., 81., 10.), color='black', linewidth=0.5, labels=[False, True, True, False])
m.drawmeridians(np.arange(-180., 181., 10.), color='black', linewidth=0.5, labels=[True, False, True, False])

x, y = m(MomentTensors[:, 4], MomentTensors[:, 3])
depth = MomentTensors[:, 5]

cmap = plt.cm.rainbow
norm = Normalize(vmin=np.min(depth), vmax=np.max(depth))

m.scatter(x, y, marker='o', c=depth, alpha=0.6, zorder=50)#edgecolor='#a2bde9')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
plt.colorbar(sm)
plt.show()

stop

m = Basemap(  #width=12000000, height=9000000,
            rsphere=(6378137.00, 6356752.3142),
            resolution='h', area_thresh=100., projection='lcc',
            # lat_1=50., lat_2=60, lat_0=55, lon_0=-135.,
            lat_0=63, lon_0=-147.,
            llcrnrlat=45, llcrnrlon=-180, urcrnrlat=65, urcrnrlon=-100,
            epsg=3338)
m.drawcoastlines()
# m.fillcontinents()
# m.drawmapboundary(fill_color='aqua')
# m.etopo()
m.arcgisimage(service='Ocean_Basemap', xpixels=20000, verbose=True)
# m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels=20000, verbose=True)
m.drawparallels(np.arange(-80., 81., 10.), color='white', linewidth=0.5, labels=[False, True, True, False])
m.drawmeridians(np.arange(-180., 181., 10.), color='white', linewidth=0.5, labels=[True, False, True, False])
# draw_screen_poly([MT.lon for MT in accepted_tensors], [MT.lat for MT in accepted_tensors], m)
# pos0 = utilities.cart2spherical((X[0, 0], Y[0, 0], Z[0, 0]))
# pos1 = utilities.cart2spherical((X[-1, 0], Y[-1, 0], Z[-1, 0]))
# pos2 = utilities.cart2spherical((X[0, -1], Y[0, -1], Z[0, -1]))
# pos3 = utilities.cart2spherical((X[-1, -1], Y[-1, -1], Z[-1, -1]))
# lats = [pos0[1], pos1[1], pos2[1], pos3[1]]
# lons = [pos0[-1], pos1[-1], pos2[-1], pos3[-1]]
#     rll = [utilities.cart2spherical((X.flatten()[i], Y.flatten()[i], Z.flatten()[i])) for i in range(len(X.flatten()))]
#     lats = [pos[1] for pos in rll]
#     lons = [pos[-1] for pos in rll]
# draw_screen_poly(lons, lats, m)
#     x, y = m(lons, lats)
#     m.scatter(x, y, marker='o', color='b', alpha=0.3)
#     x, y = m([MT.lon for MT in accepted_tensors], [MT.lat for MT in accepted_tensors])
#     m.scatter(x, y, marker='o', color='g', alpha=0.5)
#     closest_x, closest_y = m(closest_tensor.lon, closest_tensor.lat)
#     m.scatter(closest_x, closest_y, marker='D')
#     x, y = m([MT.lon for MT in rejected_tensors], [MT.lat for MT in rejected_tensors])
#     m.scatter(x, y, marker='o', color='r', alpha=0.5)
x, y = m([MT.lon for MT in MomentTensors], [MT.lat for MT in MomentTensors])
depth = [MT.depth for MT in MomentTensors]
focmecs = [MT.mt6 for MT in MomentTensors]

cmap = plt.cm.rainbow
norm = Normalize(vmin=np.min(depth), vmax=np.max(depth))

ax = plt.gca()
for i in range(len(focmecs)):
    b = beach(focmecs[i], xy=(x[i], y[i]), width=10000, facecolor=cmap(norm(depth[i])), linewidth=1)
    b.set_zorder(10)
    ax.add_collection(b)
# m.scatter(x, y, marker='o', c=depth, edgecolors='white)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
plt.colorbar(sm)

plt.show()
