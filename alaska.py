import pandas as pd
import numpy as np
import utilities
import matplotlib.pyplot as plt
from obspy.imaging.beachball import beach
import os

os.environ["PROJ_LIB"] = "C:\\Users\\Dundo\\Anaconda3\\Library\\share";  # fixr
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

df = pd.read_pickle('gcmt_lat5075_lon130180.pkl')
# print(df)

mat = df.to_numpy()

mts_6 = np.transpose(np.array([mat[:, 10], mat[:, 12], mat[:, 14], mat[:, 16], mat[:, 18], mat[:, 20]]))
iexp = mat[:, 9]
iexp = np.reshape(np.repeat(iexp, 6), np.shape(mts_6))
mts_6 = mts_6 * iexp

lat = mat[:, 1]
lon = mat[:, 2]
# mts_6 =

# setup lambert conformal basemap.
# lat_1 is first standard parallel.
# lat_2 is second standard parallel (defaults to lat_1).
# lon_0,lat_0 is central point.
# rsphere=(6378137.00,6356752.3142) specifies WGS84 ellipsoid
# area_thresh=1000 means don't plot coastline features less
# than 1000 km^2 in area.
m = Basemap(#width=12000000, height=9000000,
            rsphere=(6378137.00, 6356752.3142),
            resolution='i', area_thresh=100., projection='lcc',
            lat_1=55., lat_2=65, lat_0=62, lon_0=-158.,
            llcrnrlat=45, llcrnrlon=-180, urcrnrlat=72, urcrnrlon=-115)
m.drawcoastlines()
# m.fillcontinents(color='coral', lake_color='aqua')
# draw parallels and meridians.
m.drawparallels(np.arange(-80., 81., 20.))
m.drawmeridians(np.arange(-180., 181., 20.))
# m.drawmapboundary(fill_color='aqua')
# draw tissot's indicatrix to show distortion.
ax = plt.gca()
# for y in np.linspace(m.ymax/20,19*m.ymax/20,9):
#     for x in np.linspace(m.xmax/20,19*m.xmax/20,12):
#         lon, lat = m(x,y,inverse=True)
#         poly = m.tissot(lon,lat,1.5,100,\
#                         facecolor='green',zorder=10,alpha=0.5)
x, y = m(lon, lat)

m.scatter(x, y, marker='o', color='r', alpha=0.5)

plt.title("Lambert Conformal Projection")
plt.show()

# beaches = [beach(mts_6[i], xy=(lat[i], lon[i]), width=5) for i in range(len(mts_6))]

# event1 = MT([4.810,  -4.523,  -0.287,   4.885,   3.476, -2.651], 24)

# print(event1.mt)

# np1 = [150, 87, 1]
# mt = [-2.39, 1.04, 1.35, 0.57, -2.94, -0.94]
# beach1 = beach(np1, xy=(-70, 80), width=30)
# beach2 = beach(mts_6[5], xy=(-150, 50), width=5)
# plt.plot([-180, -130], [45, 80], "rv", ms=20)
# ax = plt.gca()
# # ax.add_collection(beach1)
# ax.add_collection(beach2)
# # for i in range(50):
# #     ax.add_collection(beaches[i])
# ax.set_aspect("equal")
# # ax.set_xlim((-180, -130))
# # ax.set_ylim((45, 80))
# plt.show()
